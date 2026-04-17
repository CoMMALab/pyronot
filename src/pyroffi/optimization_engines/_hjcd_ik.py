"""Hamiltonian Jacobian Coordinate Descent IK Solver (HJCD-IK).

Two-phase inverse kinematics solver:
  Phase 1 (Coarse):  B random seeds refined via
                       Hamiltonian coordinate-descent dynamics.
  Phase 2 (Refine):  Top-K solutions replicated with small perturbations and
                     refined via Levenberg-Marquardt optimisation.

Improvements over the naive version (matching and extending the reference CUDA):
  - Adaptive pos/ori weighting     — row-equilibrates the Jacobian so position
                                     convergence is prioritised when far away
  - Jacobi column scaling in LM    — prevents ill-conditioning when joint
                                     sensitivities differ by orders of magnitude
  - Soft joint-limit prior in LM   — quadratic penalty pulling q toward range
                                     centres; enters normal equations directly
  - Vectorised line search in LM   — 5 step-size candidates evaluated in
                                     parallel via vmap (JAX-native advantage)
  - Stall detection + random kicks — escape plateaus; λ reset after each kick
  - Best-config tracking in LM     — kicks can never degrade returned quality
  - continuity_weight selection-only — used only in the final winner pick,
                                       NOT inside optimisation (fixes dancing)

Key settings (matching the reference CUDA implementation):
  epsilon     = 20 mm  — position convergence threshold for coarse phase
  nu          = π/2    — orientation convergence threshold for coarse phase
  k_max       = 20     — coordinate-descent iteration budget
  lambda_init = 5e-3   — initial LM damping factor
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ._ik_primitives import _LS_ALPHAS, _ik_residual, _adaptive_weights, split_cuda_and_post_constraints  # noqa: F401
from ._ls_ik import _prepare_ls_collision_buffers

# Consecutive non-improving LM steps before a random kick is applied.
_STALL_PATIENCE: int = 6


# ---------------------------------------------------------------------------
# Phase 1 — Hamiltonian coordinate descent
# ---------------------------------------------------------------------------

def _hamiltonian_cd_step(
    cfg: Float[Array, "n_act"],
    momentum: Float[Array, "n_act"],
    robot: Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
    lower: Float[Array, "n_act"],
    upper: Float[Array, "n_act"],
    fixed_joint_mask: Float[Array, "n_act"],
    step_size: float = 0.35,
    momentum_decay: float = 0.9,
) -> tuple[Float[Array, "n_act"], Float[Array, "n_act"]]:
    """Single Hamiltonian coordinate-descent update.

    Uses a per-joint momentum state p and applies a one-coordinate leapfrog-like
    update on the joint with the largest normalised gradient magnitude.
    """
    # Use reverse-mode AD: 6 backward passes (one per residual dim) vs
    # n_act forward passes with jacfwd.  The primal f is obtained for free,
    # eliminating the separate _ik_residual call.
    residual_fn = lambda q: _ik_residual(q, robot, target_link_index, target_pose)
    f, vjp_fn = jax.vjp(residual_fn, cfg)
    J = jax.vmap(lambda g: vjp_fn(g)[0])(jnp.eye(6, dtype=cfg.dtype))  # (6, n_act)

    # Keep coarse-phase weighting behaviour identical across methods.
    pos_err_m = jnp.linalg.norm(f[:3])
    w_base = _adaptive_weights(f)
    ori_gate = (pos_err_m < 1e-3).astype(f.dtype)
    w = w_base * jnp.concatenate([jnp.ones(3), jnp.full(3, ori_gate)])
    Jw = J * w[:, None]
    fw = f * w

    g = jnp.einsum("ji,j->i", Jw, fw)                 # gradient-like term
    Jw_normsq = jnp.sum(Jw**2, axis=0) + 1e-8
    score = jnp.abs(g) / jnp.sqrt(Jw_normsq)
    score = jnp.where(fixed_joint_mask, -1.0, score)
    best_joint = jnp.argmax(score)

    p_best = momentum_decay * momentum[best_joint] - step_size * g[best_joint] / Jw_normsq[best_joint]
    new_momentum = momentum.at[best_joint].set(p_best)

    delta = jnp.zeros_like(cfg).at[best_joint].set(new_momentum[best_joint])
    delta = jnp.where(fixed_joint_mask, 0.0, delta)
    new_cfg = jnp.clip(cfg + delta, lower, upper)
    return new_cfg, new_momentum


@functools.partial(jax.jit, static_argnames=("target_link_index", "k_max"))
def _coarse_search_single(
    cfg: Float[Array, "n_act"],
    robot: Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
    k_max: int,
    epsilon: float,
    nu: float,
    lower: Float[Array, "n_act"],
    upper: Float[Array, "n_act"],
    fixed_joint_mask: Float[Array, "n_act"],
) -> Float[Array, "n_act"]:
    """Run k_max Hamiltonian coarse steps on one seed."""
    def body(_, carry):
        c, m = carry
        f = _ik_residual(c, robot, target_link_index, target_pose)
        done = (jnp.linalg.norm(f[:3]) < epsilon) & (jnp.linalg.norm(f[3:]) < nu)

        def _no_op(_):
            return c, m

        def _do_step(_):
            return _hamiltonian_cd_step(
                c, m, robot, target_link_index, target_pose, lower, upper, fixed_joint_mask
            )

        return jax.lax.cond(done, _no_op, _do_step, operand=None)

    momentum0 = jnp.zeros_like(cfg)
    cfg_out, _ = jax.lax.fori_loop(0, k_max, body, (cfg, momentum0))
    return cfg_out


# ---------------------------------------------------------------------------
# Phase 2 — Levenberg-Marquardt refinement
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("target_link_indices", "max_iter", "constraint_fns"),
)
def _lm_refine_single(
    cfg: Float[Array, "n_act"],
    robot: Robot,
    target_link_indices: tuple[int, ...],
    target_poses: tuple,
    max_iter: int,
    lambda_init: float,
    limit_prior_weight: float,
    kick_scale: float,
    rng_key: Array,
    lower: Float[Array, "n_act"],
    upper: Float[Array, "n_act"],
    eps_pos: float = 1e-8,
    eps_ori: float = 1e-8,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    constraint_fns: tuple = (),
    constraint_args: tuple = (),
    constraint_weights: Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    """Robust Levenberg-Marquardt refinement for one seed.

    Multi-EE support
        The residual vector becomes
        ``[w_sg_0*f_0 | ... | w_sg_{N-1}*f_{N-1} | sqrt_wc*f_coll]``.
        Each EE gets its own per-EE adaptive weight ``w_sg_i`` so a
        converged EE does not suppress orientation correction for other EEs.

    Jacobi column scaling
        Each Jacobian column is normalised to unit length before forming the
        normal equations.  This prevents ill-conditioning when joint
        sensitivities (e.g. a prismatic joint vs a shoulder rotation) differ
        by orders of magnitude.  The step is unscaled before application.

    Soft joint-limit prior
        Adds  limit_prior_weight · ‖(q − mid) / half_range‖²  to the cost.
        In the normal equations this contributes a diagonal regulariser
        D_prior and a gradient term g_prior, both transformed to the
        column-scaled space before the solve.

    Adaptive residual weighting
        `_adaptive_weights` scales orientation residuals by pos_err/ori_err
        so the solver focuses on closing the translational gap first.
        Applied to the FIRST EE only; remaining EEs use the same w_sg.

    Vectorised line search
        Five step multipliers [1, 0.5, 0.25, 0.1, 0.025] are evaluated
        in parallel via vmap (a JAX advantage over sequential CUDA backtrack).
        The multiplier that produces the lowest unweighted task error is kept.

    Stall detection + random kicks
        After _STALL_PATIENCE consecutive non-improving steps a Gaussian
        perturbation (σ = kick_scale) is applied and λ is reset to
        lambda_init.  This mirrors the CUDA stall-recovery mechanism.

    Best-configuration tracking
        The carry stores the all-time lowest-error configuration seen during
        the scan.  Random kicks therefore can never degrade the returned
        result.

    Kinematic constraints
        When ``constraint_fns`` is non-empty, pseudo-residuals
        ``sqrt(w_i) * c_i(q)`` are appended to the task residual so that
        the LM normal equations simultaneously minimise the task error and
        the weighted constraint penalties.  The acceptance criterion and
        best-config tracking both use the augmented cost.
    """
    n          = cfg.shape[0]
    n_c        = len(constraint_fns)
    n_ee       = len(target_link_indices)
    mid        = (lower + upper) * 0.5
    half_range = (upper - lower) * 0.5 + 1e-8

    # Joint-limit prior Hessian diagonal (original parameter space).
    D_prior_raw = limit_prior_weight / half_range ** 2   # (n,)
    lam0        = jnp.asarray(lambda_init, dtype=cfg.dtype)
    if fixed_joint_mask is None:
        fixed_joint_mask = jnp.zeros(n, dtype=jnp.bool_)

    if n_c > 0:
        sqrt_wc = jnp.sqrt(constraint_weights)

    def lm_step(carry, _):
        c, lam, stall_count, key, best_c, best_err = carry

        # ── Forward pass: per-EE adaptive weights + trust-region errors ──
        # Each EE gets its own adaptive weight so a converged EE (e.g. right
        # arm) does not suppress orientation correction for unconverged EEs.
        f_ee_list = [
            _ik_residual(c, robot, target_link_indices[i], target_poses[i])
            for i in range(n_ee)
        ]
        w_sg_list = [
            jax.lax.stop_gradient(_adaptive_weights(f_ee_list[i]))
            for i in range(n_ee)
        ]

        # ── Fused Jacobian: all EEs + constraints (single FK via XLA CSE) ─
        # Each EE uses its own adaptive weight (w_sg_list[i]) so a converged
        # EE doesn't suppress orientation correction for other EEs.
        # XLA CSE deduplicates the robot.forward_kinematics(q) calls across
        # the f_ee_list above and the _ik_residual calls here.
        def fused_fixed(q: Array) -> Array:
            """Per-EE weighted [EE residuals... | constraint residuals]."""
            parts: list[Array] = []
            for i in range(n_ee):
                ft = _ik_residual(q, robot, target_link_indices[i], target_poses[i])
                parts.append(ft * w_sg_list[i])
            if n_c > 0:
                fc = jnp.stack([
                    constraint_fns[i](q, robot, constraint_args[i])
                    for i in range(n_c)
                ])
                parts.append(sqrt_wc * fc)
            return jnp.concatenate(parts)

        n_out = 6 * n_ee + n_c
        f_all, vjp_fn = jax.vjp(fused_fixed, c)
        J_all  = jax.vmap(lambda g: vjp_fn(g)[0])(
            jnp.eye(n_out, dtype=f_all.dtype)
        )  # (n_out, n)
        fw_eff   = f_all
        Jw_eff   = J_all
        curr_err = jnp.dot(f_all, f_all)

        # ── Jacobi column scaling ─────────────────────────────────────────
        col_scale = jnp.linalg.norm(Jw_eff, axis=0) + 1e-8    # (n,)
        Js        = Jw_eff / col_scale[None, :]                # (n_out, n) unit-cols

        # ── Normal equations with joint-limit prior (float32) ────────────
        D_prior_s = D_prior_raw / col_scale ** 2          # (n,)
        g_prior_s = D_prior_raw * (c - mid) / col_scale   # (n,)

        A_s   = Js.T @ Js + jnp.eye(n, dtype=Js.dtype) * (lam + D_prior_s)
        rhs_s = -(Js.T @ fw_eff + g_prior_s)
        p     = jnp.linalg.solve(A_s, rhs_s)              # (n,) scaled, float32
        delta = p / col_scale                               # (n,) unscaled
        delta = jnp.where(fixed_joint_mask, 0.0, delta)   # freeze fixed joints

        # ── Trust-region step clipping ───────────────────────────────────
        # Use the MAX unweighted error across all EEs. f_ee_list[i] was
        # computed above (XLA CSE deduplicates with fused_fixed's FK calls).
        pos_err_r = jnp.max(jnp.array([
            jnp.linalg.norm(f_ee_list[i][:3]) for i in range(n_ee)
        ]))
        ori_err_r = jnp.max(jnp.array([
            jnp.linalg.norm(f_ee_list[i][3:]) for i in range(n_ee)
        ]))
        R = jnp.where(
            (pos_err_r > 1e-2) | (ori_err_r > 0.6),  0.38,
            jnp.where(
                (pos_err_r > 1e-3) | (ori_err_r > 0.25), 0.22,
                jnp.where((pos_err_r > 2e-4) | (ori_err_r > 0.08), 0.12, 0.05)
            )
        )
        delta_norm = jnp.linalg.norm(delta) + 1e-18
        delta = jnp.where(delta_norm > R, delta * R / delta_norm, delta)

        # ── Vectorised line search (fused: single FK per candidate) ──────
        def eval_alpha(alpha):
            nc     = jnp.clip(c + alpha * delta, lower, upper)
            nf_all = fused_fixed(nc)
            return jnp.dot(nf_all, nf_all)

        alpha_errs  = jax.vmap(eval_alpha)(_LS_ALPHAS)    # (5,)
        best_ls_idx = jnp.argmin(alpha_errs)
        new_err     = alpha_errs[best_ls_idx]
        new_c       = jnp.clip(c + _LS_ALPHAS[best_ls_idx] * delta, lower, upper).astype(c.dtype)

        # ── Accept / reject ───────────────────────────────────────────────
        improved = new_err < curr_err * (1.0 - 1e-4)
        c_step   = jnp.where(improved, new_c, c)
        lam_out  = jnp.clip(
            jnp.where(improved, lam * 0.5, lam * 3.0), 1e-10, 1e6
        )

        # ── Track all-time best ───────────────────────────────────────────
        new_best_c   = jnp.where(new_err < best_err, new_c,   best_c)
        new_best_err = jnp.where(new_err < best_err, new_err, best_err)

        # ── Stall detection + random kick ─────────────────────────────────
        new_stall = jnp.where(improved, jnp.zeros_like(stall_count), stall_count + 1)
        stalled   = new_stall >= _STALL_PATIENCE

        key, subkey = jax.random.split(key)
        kick        = (jax.random.normal(subkey, c.shape) * kick_scale).astype(c.dtype)
        kick        = jnp.where(fixed_joint_mask, 0.0, kick)
        c_out       = jnp.where(stalled, jnp.clip(c_step + kick, lower, upper), c_step)
        lam_out     = jnp.where(stalled, lam0, lam_out)
        stall_out   = jnp.where(stalled, jnp.zeros_like(new_stall), new_stall)

        return (c_out.astype(c.dtype), lam_out, stall_out, key, new_best_c, new_best_err), None

    # ── Initial error (same weighted metric as curr_err / eval_alpha) ────────
    # Per-EE adaptive weights, matching the lm_step convention.
    init_f_ee = [_ik_residual(cfg, robot, target_link_indices[i], target_poses[i]) for i in range(n_ee)]
    init_w_ee = [jax.lax.stop_gradient(_adaptive_weights(init_f_ee[i])) for i in range(n_ee)]

    init_parts: list[Array] = [init_f_ee[i] * init_w_ee[i] for i in range(n_ee)]
    if n_c > 0:
        sqrt_wc_init = jnp.sqrt(constraint_weights)
        init_rc = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(n_c)])
        init_parts.append(sqrt_wc_init * init_rc)
    init_fused = jnp.concatenate(init_parts)
    init_err   = jnp.dot(init_fused, init_fused)

    init_carry = (
        cfg,
        lam0,
        jnp.zeros((), dtype=jnp.int32),
        rng_key,
        cfg,       # best_c
        init_err,  # best_err
    )

    (_, _, _, _, best_c, _), _ = jax.lax.scan(
        lm_step, init_carry, None, length=max_iter
    )
    return best_c


# ---------------------------------------------------------------------------
# Adaptive refinement schedule
# ---------------------------------------------------------------------------

def _get_refine_schedule(num_seeds: int) -> tuple[int, int]:
    """Return (top_k, repeats) matching the reference HJCD-IK schedule_for_B."""
    if num_seeds <= 8:
        return num_seeds, 24
    elif num_seeds <= 16:
        return max(1, num_seeds // 2), 16
    elif num_seeds <= 128:
        return max(1, int(num_seeds * 0.2)), 5
    elif num_seeds <= 1024:
        return max(1, int(num_seeds * 0.02)), 5
    elif num_seeds <= 2048:
        return max(1, int(num_seeds * 0.01)), 5
    else:
        return max(1, int(num_seeds * 0.01)), 2


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "target_link_indices", "num_seeds", "coarse_max_iter", "lm_max_iter",
        "constraint_fns",
    ),
)
def hjcd_solve(
    robot: Robot,
    target_link_indices: tuple[int, ...],
    target_poses: tuple,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    num_seeds: int = 32,
    coarse_max_iter: int = 20,
    lm_max_iter: int = 40,
    epsilon: float = 0.02,
    nu: float = jnp.pi / 2,
    eps_pos: float = 1e-8,
    eps_ori: float = 1e-8,
    lambda_init: float = 5e-3,
    continuity_weight: float = 0.0,
    limit_prior_weight: float = 1e-4,
    kick_scale: float = 0.05,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    constraint_fns: tuple = (),
    constraint_args: tuple = (),
    constraint_weights: Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    """Solve IK via the two-phase HJCD-IK algorithm.

    Phase 1 — Coarse search
        Warm seeds near ``previous_cfg`` (σ = 0.05) are mixed with uniform
        random seeds across joint ranges.  For multi-EE problems the warm
        seeds are split into tight (σ = 0.05) and loose (σ = 0.3) halves to
        allow escape from warm-start local minima where one EE is converged
        but another is not.  All seeds then enter the coarse phase.

    Phase 2 — LM refinement
        Top-K coarse solutions are tiled ``repeats`` times with small
        Gaussian perturbations (σ = 0.01).  Each copy is refined with the
        advanced LM solver (column scaling, joint-limit prior, line search,
        stall kicks, ALL EEs in residual).  Each refinement seed gets an
        independent RNG key for its kick sequence so the batch explores
        diverse recovery trajectories.

    Multi-EE support
        Pass multiple end-effectors via ``target_link_indices`` (tuple) and
        ``target_poses`` (tuple).  For multi-EE problems the coarse-phase
        seed budget is split equally across EEs so each EE gets equal global
        exploration.  LM refinement always uses all EEs.  For a single EE,
        pass ``target_link_indices=(idx,)`` and ``target_poses=(pose,)``.

    Winner selection
        The best refined configuration is chosen by task-space residual
        summed over ALL EEs, plus ``continuity_weight`` · ‖q − previous_cfg‖².
        This weight is used **only here** (tie-breaking), never inside the
        optimisation loop, so it cannot cause the "dancing around" instability.

    Kinematic constraints
        Optional penalty terms added to the LM phase objective (Phase 2).
        Each constraint contributes ``weight_i * c_i(q)^2`` to the
        minimised cost so the solver simultaneously satisfies the task and the
        constraints.  Constraint penalties are also included in the winner
        selection score.

        Constraint functions must have signature
            ``c(cfg: Array, robot: Robot) -> Array``  (scalar)
        and should return 0 when satisfied, positive when violated.

        Note: the coarse CD phase (Phase 1) optimises the pure task objective
        only.  Constraints take effect during LM refinement.

    Args:
        robot:               The robot model.
        target_link_indices: Tuple of target link indices (static, for JIT).
                             Use a 1-tuple for single-EE problems.
        target_poses:        Tuple of desired SE(3) poses (dynamic pytree).
        rng_key:             JAX PRNG key.
        previous_cfg:        Previous joint configuration for warm-starting
                             and continuity-aware winner selection.
        num_seeds:           Coarse-phase batch size B (default 1).
        coarse_max_iter:     CD iteration budget k_max (default 20).
        lm_max_iter:         LM iterations per refinement seed (default 40).
        epsilon:             Position convergence threshold [m] for coarse phase
                             (default 20 mm).
        nu:                  Orientation convergence threshold [rad] for coarse phase
                             (default π/2).
        eps_pos:             Position convergence tolerance [m] for LM refinement;
                             iterations stop early once the best solution satisfies
                             both eps_pos and eps_ori (default 1e-8 m).
        eps_ori:             Orientation convergence tolerance [rad] for LM
                             refinement (default 1e-8 rad).
        lambda_init:         Initial LM damping factor (default 5e-3).
        continuity_weight:   Weight on ‖q − prev‖² in winner selection only
                             (default 0.0 for independent targets).
        limit_prior_weight:  Strength of the soft joint-limit prior in LM
                             (default 1e-4).
        kick_scale:          Std-dev of random kicks in LM stall recovery
                             (default 0.05 rad/m).
        fixed_joint_mask:    Boolean mask; True = joint must not move.
        constraint_fns:      Tuple of constraint callables
                             ``c(cfg, robot) -> scalar``.  Must be a ``tuple``
                             for JAX JIT compatibility; pass as
                             ``constraint_fns=tuple(my_list)``.
        constraint_weights:  Weight for each constraint, shape
                             ``(len(constraint_fns),)``.  Must be provided when
                             ``constraint_fns`` is non-empty.

    Returns:
        Best joint configuration found, shape ``(n_actuated_joints,)``.
    """
    n_act  = robot.joints.num_actuated_joints
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    if fixed_joint_mask is None:
        fixed_joint_mask = jnp.zeros(n_act, dtype=jnp.bool_)

    top_k, repeats = _get_refine_schedule(num_seeds)
    n_warm   = min(top_k, num_seeds)
    n_random = num_seeds - n_warm

    # ── Phase 1: coarse coordinate descent ────────────────────────────────
    key_seeds, key_warm, key_perturb, key_lm = jax.random.split(rng_key, 4)

    # For multi-EE problems split warm seeds into tight (σ=0.05) and loose
    # (σ=0.3) halves so some seeds can escape configurations where one EE
    # is converged but another is stuck in a warm-start local basin.
    if len(target_link_indices) > 1:
        n_tight    = max(1, n_warm // 2)
        n_loose    = n_warm - n_tight
        key_tight, key_loose = jax.random.split(key_warm)
        tight_seeds = jax.vmap(
            lambda k: jnp.clip(
                previous_cfg + jax.random.normal(k, (n_act,)) * 0.05, lower, upper
            )
        )(jax.random.split(key_tight, n_tight))
        loose_seeds = jax.vmap(
            lambda k: jnp.clip(
                previous_cfg + jax.random.normal(k, (n_act,)) * 0.3, lower, upper
            )
        )(jax.random.split(key_loose, n_loose))
        warm_seeds = jnp.concatenate([tight_seeds, loose_seeds], axis=0)
    else:
        warm_keys  = jax.random.split(key_warm, n_warm)
        warm_seeds = jax.vmap(
            lambda k: jnp.clip(
                previous_cfg + jax.random.normal(k, (n_act,)) * 0.05, lower, upper
            )
        )(warm_keys)                                           # (n_warm, n_act)
    warm_seeds = jnp.where(fixed_joint_mask[None, :], previous_cfg[None, :], warm_seeds)

    random_seeds = jax.random.uniform(
        key_seeds, (n_random, n_act), minval=lower, maxval=upper
    )                                                          # (n_random, n_act)
    random_seeds = jnp.where(fixed_joint_mask[None, :], previous_cfg[None, :], random_seeds)

    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=0)  # (B, n_act)

    # Coarse phase: distribute seeds equally across all EEs for balanced
    # global exploration.  Single-EE problems keep the original behaviour.
    # For n_ee > 1 the i-th slice of seeds does coarse search for EE i,
    # so each EE gets an equal share of the global exploration budget.
    # The per-EE compiled functions are traced at JIT time (target_link_indices
    # is static), so there is no Python-level dispatch at runtime.
    if len(target_link_indices) == 1:
        coarse_cfgs = jax.vmap(
            lambda cfg: _coarse_search_single(
                cfg, robot, target_link_indices[0], target_poses[0], coarse_max_iter,
                epsilon, nu, lower, upper,
                fixed_joint_mask,
            )
        )(seeds)                                                   # (B, n_act)
    else:
        _n_ee   = len(target_link_indices)
        _parts  = []
        for _i in range(_n_ee):
            _start = (_i * num_seeds) // _n_ee
            _end   = ((_i + 1) * num_seeds) // _n_ee
            _parts.append(jax.vmap(
                lambda cfg, _ee_idx=_i: _coarse_search_single(
                    cfg, robot, target_link_indices[_ee_idx], target_poses[_ee_idx],
                    coarse_max_iter, epsilon, nu, lower, upper, fixed_joint_mask,
                )
            )(seeds[_start:_end]))
        coarse_cfgs = jnp.concatenate(_parts, axis=0)             # (B, n_act)

    # Score coarse seeds by summing errors across ALL EEs so the top-K
    # selection is not biased toward only the first end-effector.
    coarse_errors = jax.vmap(
        lambda cfg: sum(
            jnp.sum(_ik_residual(cfg, robot, target_link_indices[i], target_poses[i]) ** 2)
            for i in range(len(target_link_indices))
        )
    )(coarse_cfgs)                                             # (B,)

    # ── Phase 2: select top-K, perturb, refine with LM (ALL EEs) ──────────
    top_k_indices = jnp.argsort(coarse_errors)[:top_k]
    top_k_cfgs    = coarse_cfgs[top_k_indices]                # (top_k, n_act)

    base_cfgs    = jnp.tile(top_k_cfgs, (repeats, 1))         # (top_k*repeats, n_act)
    perturb_keys = jax.random.split(key_perturb, top_k * repeats)
    noise        = jax.vmap(
        lambda k: jax.random.normal(k, (n_act,)) * 0.15 * (upper - lower)
    )(perturb_keys)
    noise        = jnp.where(fixed_joint_mask[None, :], 0.0, noise)
    is_original  = (jnp.arange(top_k * repeats) < top_k)[:, None]
    refine_seeds = jnp.where(
        is_original, base_cfgs, jnp.clip(base_cfgs + noise, lower, upper)
    )

    # Independent RNG keys for stall-kick noise in each LM run.
    lm_keys = jax.random.split(key_lm, top_k * repeats)

    refine_cfgs = jax.vmap(
        lambda cfg, key: _lm_refine_single(
            cfg, robot, target_link_indices, target_poses,
            lm_max_iter, lambda_init, limit_prior_weight, kick_scale,
            key, lower, upper, eps_pos, eps_ori, fixed_joint_mask,
            constraint_fns=constraint_fns,
            constraint_args=constraint_args,
            constraint_weights=constraint_weights,
        )
    )(refine_seeds, lm_keys)                                   # (top_k*repeats, n_act)

    # ── Winner selection: task error (all EEs) + constraint penalties + continuity
    # continuity_weight acts only here, NOT during optimisation.
    def winner_err(cfg):
        # Sum squared residuals over all EEs.
        task_err = sum(
            jnp.sum(_ik_residual(cfg, robot, target_link_indices[i], target_poses[i]) ** 2)
            for i in range(len(target_link_indices))
        )
        if len(constraint_fns) > 0:
            c_vals = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(len(constraint_fns))])
            task_err = task_err + jnp.sum(constraint_weights * c_vals ** 2)
        return task_err + continuity_weight * jnp.sum((cfg - previous_cfg) ** 2)

    refine_errors = jax.vmap(winner_err)(refine_cfgs)

    best_idx = jnp.argmin(refine_errors)
    return refine_cfgs[best_idx]


# ---------------------------------------------------------------------------
# CUDA solver
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "coarse_max_iter",
        "lm_max_iter",
        "lambda_init",
        "limit_prior_weight",
        "kick_scale",
        "eps_pos",
        "eps_ori",
        "enable_collision",
        "collision_weight",
        "collision_margin",
        "constraint_fns",
        "target_link_indices",
    ),
)
def _hjcd_solve_cuda_jit(
    robot: Robot,
    target_poses: tuple,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    num_seeds: int,
    coarse_max_iter: int,
    lm_max_iter: int,
    epsilon: float,
    nu: float,
    eps_pos: float,
    eps_ori: float,
    lambda_init: float,
    continuity_weight: float,
    limit_prior_weight: float,
    kick_scale: float,
    fixed_joint_mask_int: Float[Array, "n_act"],
    ancestor_masks: Array,
    target_jnts: Array,
    robot_spheres_local:  Float[Array, "n_rs 4"],
    robot_sphere_joint_idx: Array,
    world_spheres:        Float[Array, "n_ws 4"],
    world_capsules:       Float[Array, "n_wc 7"],
    world_boxes:          Float[Array, "n_wb 15"],
    world_halfspaces:     Float[Array, "n_wh 6"],
    enable_collision:     bool,
    collision_weight:     float,
    collision_margin:     float,
    target_link_indices: tuple[int, ...],
    constraint_fns: tuple = (),
    constraint_args: tuple = (),
    constraint_weights: Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    from ..cuda_kernels._hjcd_ik_cuda import hjcd_ik_coarse_cuda, hjcd_ik_lm_cuda

    n_act  = robot.joints.num_actuated_joints
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    # Stack all EE target poses: (n_ee, 7).
    target_Ts = jnp.stack([tp.wxyz_xyz.astype(jnp.float32) for tp in target_poses], axis=0)

    # ── Seed generation (mirrors hjcd_solve exactly) ──────────────────────
    top_k, repeats = _get_refine_schedule(num_seeds)
    n_warm   = min(top_k, num_seeds)
    n_random = num_seeds - n_warm

    key_seeds, key_warm, key_perturb, key_lm = jax.random.split(rng_key, 4)

    # Batched seed generation (fewer JAX dispatches than split+vmap).
    warm_seeds = jnp.clip(
        previous_cfg[None, :] + jax.random.normal(key_warm, (n_warm, n_act)) * 0.05,
        lower, upper,
    )
    warm_seeds = jnp.where(fixed_joint_mask_int[None, :], previous_cfg[None, :], warm_seeds)

    random_seeds = jax.random.uniform(key_seeds, (n_random, n_act), minval=lower, maxval=upper)
    random_seeds = jnp.where(fixed_joint_mask_int[None, :], previous_cfg[None, :], random_seeds)

    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=0)  # (num_seeds, n_act)

    # ── Phase 1: CUDA coarse coordinate descent (all EEs) ────────────────
    coarse_cfgs, coarse_errors = hjcd_ik_coarse_cuda(
        seeds=seeds[None],              # (1, num_seeds, n_act)
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
        ancestor_masks=ancestor_masks,
        target_T=target_Ts[None],       # (1, n_ee, 7)
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_joint_mask_int,
        target_jnts=target_jnts,
        k_max=coarse_max_iter,
        enable_collision=enable_collision,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
    )
    coarse_cfgs   = coarse_cfgs[0]    # (num_seeds, n_act)
    coarse_errors = coarse_errors[0]  # (num_seeds,) — all EEs from CUDA kernel

    # ── Phase 2 setup: top-K selection + perturbation ─────────────────────
    # CUDA coarse errors already cover all EEs — no JAX rescoring needed.
    _, top_k_indices = jax.lax.top_k(-coarse_errors, top_k)
    top_k_cfgs    = coarse_cfgs[top_k_indices]                    # (top_k, n_act)

    base_cfgs    = jnp.tile(top_k_cfgs, (repeats, 1))            # (top_k*repeats, n_act)
    # Batched perturbation noise (single random call instead of split+vmap).
    noise        = jax.random.normal(
        key_perturb, (top_k * repeats, n_act)
    ) * 0.15 * (upper - lower)
    noise        = jnp.where(fixed_joint_mask_int[None, :], 0.0, noise)
    is_original  = (jnp.arange(top_k * repeats) < top_k)[:, None]
    lm_seeds     = jnp.where(
        is_original, base_cfgs, jnp.clip(base_cfgs + noise, lower, upper)
    )

    # Pre-generate Gaussian kick noise for all seeds × all iterations.
    n_lm_seeds    = top_k * repeats
    lm_kick_noise = jax.random.normal(
        key_lm, (n_lm_seeds, lm_max_iter, n_act)
    ).astype(jnp.float32)

    # ── Phase 2: CUDA Levenberg-Marquardt refinement (all EEs) ───────────
    refine_cfgs, refine_errors_raw = hjcd_ik_lm_cuda(
        seeds=lm_seeds[None],           # (1, n_lm_seeds, n_act)
        noise=lm_kick_noise[None],      # (1, n_lm_seeds, lm_max_iter, n_act)
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
        ancestor_masks=ancestor_masks,
        target_T=target_Ts[None],       # (1, n_ee, 7)
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_joint_mask_int,
        target_jnts=target_jnts,
        max_iter=lm_max_iter,
        stall_patience=_STALL_PATIENCE,
        lambda_init=lambda_init,
        limit_prior_weight=limit_prior_weight,
        kick_scale=kick_scale,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        enable_collision=enable_collision,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
    )
    refine_cfgs       = refine_cfgs[0]       # (n_lm_seeds, n_act)
    refine_errors_raw = refine_errors_raw[0]  # (n_lm_seeds,) — all EEs from CUDA

    # CUDA LM errors already cover all EEs — no JAX rescoring needed.
    base_errors = refine_errors_raw

    if len(constraint_fns) > 0:
        def constraint_penalty(cfg):
            c_vals = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(len(constraint_fns))])
            return jnp.sum(constraint_weights * c_vals ** 2)

        constraint_errors = jax.vmap(constraint_penalty)(refine_cfgs)  # (n_lm_seeds,)
        refine_errors = (
            base_errors
            + constraint_errors
            + continuity_weight * jnp.sum((refine_cfgs - previous_cfg) ** 2, axis=-1)
        )
    else:
        refine_errors = (
            base_errors
            + continuity_weight * jnp.sum((refine_cfgs - previous_cfg) ** 2, axis=-1)
        )

    best_idx = jnp.argmin(refine_errors)
    return refine_cfgs[best_idx]


def hjcd_solve_cuda(
    robot: Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses: jaxlie.SE3 | tuple,
    rng_key: Array,
    previous_cfg: Float[Array, "n_act"],
    num_seeds: int = 1024,
    coarse_max_iter: int = 20,
    lm_max_iter: int = 40,
    epsilon: float = 0.02,
    nu: float = float(jnp.pi / 2),
    eps_pos: float = 1e-8,
    eps_ori: float = 1e-8,
    lambda_init: float = 5e-3,
    continuity_weight: float = 0.0,
    limit_prior_weight: float = 1e-3,
    kick_scale: float = 0.015,
    fixed_joint_mask: Float[Array, "n_act"] | None = None,
    constraints: Sequence[Callable] | None = None,
    constraint_args: Sequence | None = None,
    constraint_weights: Sequence[float] | None = None,
    collision_constraint_indices: Sequence[int] | None = None,
    collision_free: bool = False,
    collision_checker: Any | None = None,
    collision_world: Any | None = None,
    collision_weight: float = 1e4,
    collision_margin: float = 0.02,
    constraint_refine_iters: int = 12,
) -> Float[Array, "n_act"]:
    """CUDA alternative to :func:`hjcd_solve`.

    Implements the same two-phase HJCD-IK algorithm but offloads the
    coordinate-descent and Levenberg-Marquardt loops to CUDA kernels via
    the JAX FFI.  The kernels are defined in ``_hjcd_ik_cuda_kernel.cu`` and
    reuse the FK device function from ``_fk_cuda_helpers.cuh``.

    This entry point uses a partial JIT wrapper for the JAX-side logic while
    keeping Python-level precomputation (ancestor mask) outside the trace.
    The CUDA kernels are still launched via ``jax.ffi.ffi_call`` inside the
    JITed region, so seed generation, top-K selection, and winner selection
    benefit from JAX's tracing/dispatch cache.

    Multi-EE support
        Pass multiple end-effectors via ``target_link_indices`` (tuple) and
        ``target_poses`` (tuple).  The CUDA kernels optimise only the FIRST
        EE; remaining EEs are incorporated into winner selection and
        post-CUDA JAX refinement.  For a single EE, pass
        ``target_link_indices=(idx,)`` and ``target_poses=(pose,)``.

    Key numerical improvement over the JAX path:
      - Normal equations are formed and solved in **float64** inside the CUDA
        kernel, which avoids the ill-conditioning that causes the accuracy
        issues in the float32 JAX path.
      - All kernel launches use the caller's CUDA stream so there are no
        implicit device synchronisations.

    Requires ``_hjcd_ik_cuda_lib.so`` to be compiled first:
        bash src/pyroffi/cuda_kernels/build_hjcd_ik_cuda.sh

    Kinematic constraints
        Because the CUDA kernel cannot call arbitrary Python/JAX functions,
        constraints are incorporated in two stages:

        1. **Winner selection** — constraint penalties are evaluated via JAX
           on all CUDA-returned candidates and added to the selection score.
        2. **Post-CUDA JAX refinement** — ``constraint_refine_iters`` LM steps
           are run on the selected winner using the full constraint-augmented
           JAX solver (:func:`_lm_refine_single`).  Set
           ``constraint_refine_iters=0`` to skip this pass.

    Args:
        robot:                   The robot model.
        target_link_indices:     Index (or tuple of indices) of target link(s).
        target_poses:            Desired SE(3) world pose (or tuple of poses).
        rng_key:                 JAX PRNG key.
        previous_cfg:            Previous joint configuration (warm-start + continuity).
        num_seeds:               Coarse-phase batch size B.
        coarse_max_iter:         CD iteration budget.
        lm_max_iter:             LM iterations per refinement seed.
        epsilon:                 Position convergence threshold [m] used by
                                 Hamiltonian coarse mode early-stop.
        nu:                      Orientation convergence threshold [rad] used by
                                 Hamiltonian coarse mode early-stop.
        eps_pos:                 Position convergence tolerance [m] for LM early exit.
        eps_ori:                 Orientation convergence tolerance [rad] for LM early exit.
        lambda_init:             Initial LM damping factor.
        continuity_weight:       Weight on ‖q − previous_cfg‖² in winner selection.
        limit_prior_weight:      Strength of soft joint-limit prior in LM.
        kick_scale:              Std-dev of random kicks in LM stall recovery.
        fixed_joint_mask:        Boolean mask; True = joint must not move.
        constraints:             List of constraint callables
                                 ``c(cfg, robot) -> scalar``.
        constraint_weights:      Scalar weight for each constraint.
        constraint_refine_iters: JAX LM iterations applied post-CUDA on the
                                 winner when constraints are provided (default 12).

    Returns:
        Best joint configuration found, shape ``(n_act,)``.
    """
    # Normalise scalar → 1-tuple API.
    if isinstance(target_link_indices, int):
        target_link_indices = (target_link_indices,)
    if isinstance(target_poses, jaxlie.SE3):
        target_poses = (target_poses,)
    target_poses_t = tuple(target_poses)

    n_act  = robot.joints.num_actuated_joints

    if fixed_joint_mask is None:
        fixed_joint_mask_int = jnp.zeros(n_act, dtype=jnp.int32)
    else:
        fixed_joint_mask_int = fixed_joint_mask.astype(jnp.int32)

    (
        cuda_constraint_fns,
        cuda_constraint_args,
        cuda_constraint_weights,
        post_constraint_fns,
        post_constraint_args,
        post_constraint_weights,
    ) = split_cuda_and_post_constraints(
        constraints=constraints,
        constraint_args=constraint_args,
        constraint_weights=constraint_weights,
        collision_constraint_indices=collision_constraint_indices,
        collision_free=collision_free,
    )

    # ── Pre-compute per-EE ancestor masks (Python level) ───────────────────
    # These cannot be computed inside jax.jit because parent_joint_indices
    # and parent_indices are JAX-array leaves of the Robot pytree.
    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    parent_idx_np           = np.array(robot.joints.parent_indices)
    n_joints                = robot.joints.num_joints
    n_ee_count              = len(target_link_indices)

    target_joints_np    = np.zeros(n_ee_count, dtype=np.int32)
    ancestor_masks_np   = np.zeros((n_ee_count, n_joints), dtype=np.int32)
    for _i, _link_idx in enumerate(target_link_indices):
        _tgt_jnt = int(parent_joint_indices_np[_link_idx])
        target_joints_np[_i] = _tgt_jnt
        _j = _tgt_jnt
        while _j >= 0:
            ancestor_masks_np[_i, _j] = 1
            _j = int(parent_idx_np[_j])

    target_jnts    = jnp.array(target_joints_np)
    ancestor_masks = jnp.array(ancestor_masks_np)

    (
        robot_spheres_local,
        robot_sphere_joint_idx,
        world_spheres,
        world_capsules,
        world_boxes,
        world_halfspaces,
        collision_enabled,
    ) = _prepare_ls_collision_buffers(robot, collision_checker, collision_world)

    winner = _hjcd_solve_cuda_jit(
        robot=robot,
        target_poses=target_poses_t,
        rng_key=rng_key,
        previous_cfg=previous_cfg,
        num_seeds=num_seeds,
        coarse_max_iter=coarse_max_iter,
        lm_max_iter=lm_max_iter,
        epsilon=epsilon,
        nu=nu,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        lambda_init=lambda_init,
        continuity_weight=continuity_weight,
        limit_prior_weight=limit_prior_weight,
        kick_scale=kick_scale,
        fixed_joint_mask_int=fixed_joint_mask_int,
        ancestor_masks=ancestor_masks,
        target_jnts=target_jnts,
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        enable_collision=bool(collision_free and collision_enabled),
        collision_weight=collision_weight,
        collision_margin=collision_margin,
        target_link_indices=target_link_indices,
        constraint_fns=cuda_constraint_fns,
        constraint_args=cuda_constraint_args,
        constraint_weights=cuda_constraint_weights,
    )

    # ── Post-CUDA JAX refinement with constraints only ─────────────────────
    # Multi-EE is now handled in CUDA; only constraints require JAX refinement.
    needs_refinement = bool(post_constraint_fns)
    if needs_refinement and constraint_refine_iters > 0:
        fmask = (
            fixed_joint_mask.astype(jnp.bool_)
            if fixed_joint_mask is not None
            else jnp.zeros(n_act, dtype=jnp.bool_)
        )
        key_post = jax.random.PRNGKey(0)  # deterministic post-refinement key
        winner = _lm_refine_single(
            winner, robot, target_link_indices, target_poses_t,
            constraint_refine_iters, lambda_init, limit_prior_weight, kick_scale,
            key_post, robot.joints.lower_limits, robot.joints.upper_limits,
            eps_pos, eps_ori, fmask,
            constraint_fns=post_constraint_fns,
            constraint_args=post_constraint_args,
            constraint_weights=post_constraint_weights,
        )

    return winner


# ---------------------------------------------------------------------------
# CUDA batched solver
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "coarse_max_iter",
        "lm_max_iter",
        "lambda_init",
        "limit_prior_weight",
        "kick_scale",
        "eps_pos",
        "eps_ori",
        "enable_collision",
        "collision_weight",
        "collision_margin",
        "constraint_fns",
        "target_link_indices",
    ),
)
def _hjcd_solve_cuda_batch_jit(
    robot:                Robot,
    target_poses_batch:   jaxlie.SE3,
    rng_key:              Array,
    previous_cfgs:        Float[Array, "n_problems n_act"],
    num_seeds:            int,
    coarse_max_iter:      int,
    lm_max_iter:          int,
    epsilon:              float,
    nu:                   float,
    eps_pos:              float,
    eps_ori:              float,
    lambda_init:          float,
    continuity_weight:    float,
    limit_prior_weight:   float,
    kick_scale:           float,
    fixed_joint_mask_int: Float[Array, "n_act"],
    ancestor_masks:       Array,
    target_jnts:          Array,
    robot_spheres_local:  Float[Array, "n_rs 4"],
    robot_sphere_joint_idx: Array,
    world_spheres:        Float[Array, "n_ws 4"],
    world_capsules:       Float[Array, "n_wc 7"],
    world_boxes:          Float[Array, "n_wb 15"],
    world_halfspaces:     Float[Array, "n_wh 6"],
    enable_collision:     bool,
    collision_weight:     float,
    collision_margin:     float,
    target_link_indices:  tuple[int, ...],
    constraint_fns:       tuple = (),
    constraint_args:      tuple = (),
    constraint_weights:   Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_problems n_act"]:
    from ..cuda_kernels._hjcd_ik_cuda import hjcd_ik_coarse_cuda, hjcd_ik_lm_cuda

    n_act      = robot.joints.num_actuated_joints
    lower      = robot.joints.lower_limits
    upper      = robot.joints.upper_limits
    n_problems = previous_cfgs.shape[0]

    # Batched target poses: (n_problems, 7) wrapped to (n_problems, 1, 7) for n_ee=1.
    target_T_batch = target_poses_batch.wxyz_xyz.astype(jnp.float32)[:, None, :]  # (n_problems, 1, 7)

    # ── Seed generation (mirrors hjcd_solve exactly) ──────────────────────
    top_k, repeats = _get_refine_schedule(num_seeds)
    n_warm   = min(top_k, num_seeds)
    n_random = num_seeds - n_warm

    key_seeds, key_warm, key_perturb, key_lm = jax.random.split(rng_key, 4)

    warm_seeds = jnp.clip(
        previous_cfgs[:, None, :] + jax.random.normal(key_warm, (n_problems, n_warm, n_act)) * 0.05,
        lower, upper,
    )
    warm_seeds = jnp.where(fixed_joint_mask_int[None, None, :], previous_cfgs[:, None, :], warm_seeds)

    random_seeds = jax.random.uniform(
        key_seeds, (n_problems, n_random, n_act), minval=lower, maxval=upper
    )
    random_seeds = jnp.where(
        fixed_joint_mask_int[None, None, :], previous_cfgs[:, None, :], random_seeds
    )

    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=1)  # (n_problems, num_seeds, n_act)

    # ── Phase 1: CUDA coarse coordinate descent (all EEs) ────────────────
    coarse_cfgs, coarse_errors = hjcd_ik_coarse_cuda(
        seeds=seeds,                     # (n_problems, num_seeds, n_act)
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
        ancestor_masks=ancestor_masks,
        target_T=target_T_batch,         # (n_problems, 1, 7)
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_joint_mask_int,
        target_jnts=target_jnts,
        k_max=coarse_max_iter,
        enable_collision=enable_collision,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
    )

    # ── Phase 2 setup: top-K selection + perturbation ─────────────────────
    _, top_k_indices = jax.lax.top_k(-coarse_errors, top_k)
    top_k_cfgs = jax.vmap(lambda cfgs, idx: cfgs[idx])(coarse_cfgs, top_k_indices)

    base_cfgs = jnp.tile(top_k_cfgs, (1, repeats, 1))
    noise = jax.random.normal(
        key_perturb, (n_problems, top_k * repeats, n_act)
    ) * 0.15 * (upper - lower)
    noise = jnp.where(fixed_joint_mask_int[None, None, :], 0.0, noise)
    is_original = (jnp.arange(top_k * repeats) < top_k)[None, :, None]
    lm_seeds = jnp.where(
        is_original, base_cfgs, jnp.clip(base_cfgs + noise, lower, upper)
    )

    # Pre-generate Gaussian kick noise for all problems × all seeds × all iterations.
    n_lm_seeds = top_k * repeats
    lm_kick_noise = jax.random.normal(
        key_lm, (n_problems, n_lm_seeds, lm_max_iter, n_act)
    ).astype(jnp.float32)

    # ── Phase 2: CUDA Levenberg-Marquardt refinement (all EEs) ───────────
    refine_cfgs, refine_errors_raw = hjcd_ik_lm_cuda(
        seeds=lm_seeds,
        noise=lm_kick_noise,
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
        ancestor_masks=ancestor_masks,
        target_T=target_T_batch,         # (n_problems, 1, 7)
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_joint_mask_int,
        target_jnts=target_jnts,
        max_iter=lm_max_iter,
        stall_patience=_STALL_PATIENCE,
        lambda_init=lambda_init,
        limit_prior_weight=limit_prior_weight,
        kick_scale=kick_scale,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        enable_collision=enable_collision,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
    )

    # ── Winner selection: task (all EEs) + constraint penalties + continuity ─
    if len(constraint_fns) > 0:
        flat_cfgs = refine_cfgs.reshape(n_problems * n_lm_seeds, n_act)

        def constraint_penalty(cfg):
            c_vals = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(len(constraint_fns))])
            return jnp.sum(constraint_weights * c_vals ** 2)

        flat_cpen = jax.vmap(constraint_penalty)(flat_cfgs)
        cpen      = flat_cpen.reshape(n_problems, n_lm_seeds)

        refine_errors = (
            refine_errors_raw
            + cpen
            + continuity_weight * jnp.sum((refine_cfgs - previous_cfgs[:, None, :]) ** 2, axis=-1)
        )
    else:
        refine_errors = (
            refine_errors_raw
            + continuity_weight * jnp.sum((refine_cfgs - previous_cfgs[:, None, :]) ** 2, axis=-1)
        )

    best_idx = jnp.argmin(refine_errors, axis=1)
    return refine_cfgs[jnp.arange(n_problems), best_idx]


def hjcd_solve_cuda_batch(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3,
    rng_key:             Array,
    previous_cfgs:       Float[Array, "n_problems n_act"],
    num_seeds:           int   = 32,
    coarse_max_iter:     int   = 20,
    lm_max_iter:         int   = 40,
    epsilon:             float = 0.02,
    nu:                  float = float(jnp.pi / 2),
    eps_pos:             float = 1e-8,
    eps_ori:             float = 1e-8,
    lambda_init:         float = 5e-3,
    continuity_weight:   float = 0.0,
    limit_prior_weight:  float = 1e-3,
    kick_scale:          float = 0.015,
    fixed_joint_mask:    Float[Array, "n_act"] | None = None,
    constraints:         Sequence[Callable] | None = None,
    constraint_args:     Sequence | None = None,
    constraint_weights:  Sequence[float] | None = None,
    collision_constraint_indices: Sequence[int] | None = None,
    collision_free:      bool = False,
    collision_checker:   Any | None = None,
    collision_world:     Any | None = None,
    collision_weight:    float = 1e4,
    collision_margin:    float = 0.02,
    constraint_refine_iters: int = 30,
) -> Float[Array, "n_problems n_act"]:
    """Batched CUDA HJCD-IK: solve n_problems targets in a single kernel launch.

    Kinematic constraints
        Constraints are evaluated via JAX on all returned candidates for
        winner selection.  When ``constraint_refine_iters > 0`` a short
        constraint-aware JAX LM pass is applied to each problem's winner
        (vmapped over the batch).

    Args:
        robot:                   The robot model.
        target_link_indices:     Index (or tuple of indices) of target link(s).
        target_poses:            Batch of SE(3) targets for the FIRST EE,
                                 shape ``(n_problems,)`` pytree.
        rng_key:                 JAX PRNG key.
        previous_cfgs:           Previous configurations, shape ``(n_problems, n_act)``.
        num_seeds:               Coarse-phase batch size per problem.
        coarse_max_iter:         CD iteration budget.
        lm_max_iter:             LM iterations per refinement seed.
        epsilon:                 Position convergence threshold [m] for coarse phase.
        nu:                      Orientation convergence threshold [rad] for coarse phase.
        eps_pos:                 Position convergence tolerance [m] for LM early exit.
        eps_ori:                 Orientation convergence tolerance [rad] for LM early exit.
        lambda_init:             Initial LM damping factor.
        continuity_weight:       Weight on ‖q − prev‖² in winner selection only.
        limit_prior_weight:      Strength of soft joint-limit prior in LM.
        kick_scale:              Std-dev of random kicks in LM stall recovery.
        fixed_joint_mask:        Boolean mask; True = joint must not move.
        constraints:             List of constraint callables
                                 ``c(cfg, robot) -> scalar``.
        constraint_weights:      Scalar weight for each constraint.
        constraint_refine_iters: JAX LM iterations applied post-CUDA on each
                                 problem's winner (default 30, set 0 to disable).

    Returns:
        Best joint configurations, shape ``(n_problems, n_act)``.
    """
    # Normalise scalar → 1-tuple API.
    if isinstance(target_link_indices, int):
        target_link_indices = (target_link_indices,)

    n_act      = robot.joints.num_actuated_joints

    if fixed_joint_mask is None:
        fixed_joint_mask_int = jnp.zeros(n_act, dtype=jnp.int32)
    else:
        fixed_joint_mask_int = fixed_joint_mask.astype(jnp.int32)

    (
        cuda_constraint_fns,
        cuda_constraint_args,
        cuda_constraint_weights,
        post_constraint_fns,
        post_constraint_args,
        post_constraint_weights,
    ) = split_cuda_and_post_constraints(
        constraints=constraints,
        constraint_args=constraint_args,
        constraint_weights=constraint_weights,
        collision_constraint_indices=collision_constraint_indices,
        collision_free=collision_free,
    )

    # Ancestor mask using first EE only (batch path is single-EE; wrap in n_ee=1 format).
    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    target_joint_idx = int(parent_joint_indices_np[target_link_indices[0]])
    parent_idx_np = np.array(robot.joints.parent_indices)
    n_joints = robot.joints.num_joints
    ancestor_mask_np = np.zeros(n_joints, dtype=np.int32)
    j = target_joint_idx
    while j >= 0:
        ancestor_mask_np[j] = 1
        j = int(parent_idx_np[j])
    # Wrap in (1, n_joints) for n_ee=1.
    ancestor_masks = jnp.array(ancestor_mask_np[None, :])
    target_jnts    = jnp.array([target_joint_idx], dtype=jnp.int32)

    (
        robot_spheres_local,
        robot_sphere_joint_idx,
        world_spheres,
        world_capsules,
        world_boxes,
        world_halfspaces,
        collision_enabled,
    ) = _prepare_ls_collision_buffers(robot, collision_checker, collision_world)

    winners = _hjcd_solve_cuda_batch_jit(
        robot=robot,
        target_poses_batch=target_poses,
        rng_key=rng_key,
        previous_cfgs=previous_cfgs,
        num_seeds=num_seeds,
        coarse_max_iter=coarse_max_iter,
        lm_max_iter=lm_max_iter,
        epsilon=epsilon,
        nu=nu,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        lambda_init=lambda_init,
        continuity_weight=continuity_weight,
        limit_prior_weight=limit_prior_weight,
        kick_scale=kick_scale,
        fixed_joint_mask_int=fixed_joint_mask_int,
        ancestor_masks=ancestor_masks,
        target_jnts=target_jnts,
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        enable_collision=bool(collision_free and collision_enabled),
        collision_weight=collision_weight,
        collision_margin=collision_margin,
        target_link_indices=target_link_indices,
        constraint_fns=cuda_constraint_fns,
        constraint_args=cuda_constraint_args,
        constraint_weights=cuda_constraint_weights,
    )

    # ── Post-CUDA JAX refinement with all EEs + constraints (vmapped over batch)
    if post_constraint_fns and constraint_refine_iters > 0:
        fmask = (
            fixed_joint_mask.astype(jnp.bool_)
            if fixed_joint_mask is not None
            else jnp.zeros(n_act, dtype=jnp.bool_)
        )
        lower = robot.joints.lower_limits
        upper = robot.joints.upper_limits
        key_post = jax.random.PRNGKey(0)

        winners = jax.vmap(
            lambda cfg, wxyz_xyz: _lm_refine_single(
                cfg, robot, target_link_indices,
                (jaxlie.SE3(wxyz_xyz.astype(cfg.dtype)),),
                constraint_refine_iters, lambda_init, limit_prior_weight, kick_scale,
                key_post, lower, upper, eps_pos, eps_ori, fmask,
                constraint_fns=post_constraint_fns,
                constraint_args=post_constraint_args,
                constraint_weights=post_constraint_weights,
            )
        )(winners, target_poses.wxyz_xyz)

    return winners
