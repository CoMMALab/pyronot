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

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ._ik_primitives import _LS_ALPHAS, _ik_residual, _adaptive_weights  # noqa: F401

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

@functools.partial(jax.jit, static_argnames=("target_link_index", "max_iter"))
def _lm_refine_single(
    cfg: Float[Array, "n_act"],
    robot: Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
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
) -> Float[Array, "n_act"]:
    """Robust Levenberg-Marquardt refinement for one seed.

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
    """
    n          = cfg.shape[0]
    mid        = (lower + upper) * 0.5
    half_range = (upper - lower) * 0.5 + 1e-8

    # Joint-limit prior Hessian diagonal (original parameter space).
    D_prior_raw = limit_prior_weight / half_range ** 2   # (n,)
    lam0        = jnp.asarray(lambda_init, dtype=cfg.dtype)
    if fixed_joint_mask is None:
        fixed_joint_mask = jnp.zeros(n, dtype=jnp.bool_)

    def lm_step(carry, _):
        c, lam, stall_count, key, best_c, best_err, done = carry

        # ── Check refinement convergence on all-time best ────────────────
        f_best     = _ik_residual(best_c, robot, target_link_index, target_pose)
        already_done = done | (
            (jnp.linalg.norm(f_best[:3]) < eps_pos)
            & (jnp.linalg.norm(f_best[3:]) < eps_ori)
        )

        def _do_step(inner):
            c, lam, stall_count, key, best_c, best_err = inner

            # ── Jacobian + residual ──────────────────────────────────────
            # Reverse-mode AD: 6 backward passes vs n_act forward passes.
            # The primal f comes for free, saving one extra FK evaluation.
            residual_fn = lambda q: _ik_residual(q, robot, target_link_index, target_pose)
            f, vjp_fn = jax.vjp(residual_fn, c)
            J = jax.vmap(lambda g: vjp_fn(g)[0])(jnp.eye(6, dtype=c.dtype))  # (6, n)
            curr_err = jnp.dot(f, f)

            # ── Row equilibration: adaptive pos/ori weighting ────────────
            w  = _adaptive_weights(f)
            Jw = J * w[:, None]    # (6, n)
            fw = f * w             # (6,)

            # ── Jacobi column scaling ────────────────────────────────────
            col_scale = jnp.linalg.norm(Jw, axis=0) + 1e-8    # (n,)
            Js        = Jw / col_scale[None, :]                # (6, n) unit-cols

            # ── Normal equations with joint-limit prior (scaled space) ───
            # Formed and solved in float64 to avoid ill-conditioning (matches
            # the CUDA kernel which uses double for the Cholesky solve).
            D_prior_s = D_prior_raw / col_scale ** 2          # (n,)
            g_prior_s = D_prior_raw * (c - mid) / col_scale   # (n,)

            Js_d  = Js.astype(jnp.float64)
            fw_d  = fw.astype(jnp.float64)
            lam_d = lam.astype(jnp.float64)
            D_d   = D_prior_s.astype(jnp.float64)
            g_d   = g_prior_s.astype(jnp.float64)

            A_s   = Js_d.T @ Js_d + jnp.eye(n, dtype=jnp.float64) * (lam_d + D_d)
            rhs_s = -(Js_d.T @ fw_d + g_d)
            p     = jnp.linalg.solve(A_s, rhs_s).astype(c.dtype)  # (n,) scaled
            delta = p / col_scale                               # (n,) unscaled
            delta = jnp.where(fixed_joint_mask, 0.0, delta)   # freeze fixed joints
            # ── Trust-region step clipping (matches reference radius schedule) ──
            pos_err_r = jnp.linalg.norm(f[:3])
            ori_err_r = jnp.linalg.norm(f[3:])
            R = jnp.where(
                (pos_err_r > 1e-2) | (ori_err_r > 0.6),  0.38,
                jnp.where(
                    (pos_err_r > 1e-3) | (ori_err_r > 0.25), 0.22,
                    jnp.where((pos_err_r > 2e-4) | (ori_err_r > 0.08), 0.12, 0.05)
                )
            )
            delta_norm = jnp.linalg.norm(delta) + 1e-18
            delta = jnp.where(delta_norm > R, delta * R / delta_norm, delta)
            # ── Vectorised line search ───────────────────────────────────
            def eval_alpha(alpha):
                new_c_a = jnp.clip(c + alpha * delta, lower, upper)
                new_f   = _ik_residual(new_c_a, robot, target_link_index, target_pose)
                return jnp.dot(new_f, new_f)

            alpha_errs  = jax.vmap(eval_alpha)(_LS_ALPHAS)    # (5,)
            best_ls_idx = jnp.argmin(alpha_errs)
            new_err     = alpha_errs[best_ls_idx]
            new_c       = jnp.clip(c + _LS_ALPHAS[best_ls_idx] * delta, lower, upper)

            # ── Accept / reject ──────────────────────────────────────────
            # Drift guard: require at least 1e-4 relative improvement to
            # accept.  Pure floating-point noise sits well below this
            # threshold, so the guard prevents false "improvements" from
            # perturbing the trajectory.
            improved = new_err < curr_err * (1.0 - 1e-4)
            c_step   = jnp.where(improved, new_c, c)
            lam_out  = jnp.clip(
                jnp.where(improved, lam * 0.5, lam * 3.0), 1e-10, 1e6
            )

            # ── Track all-time best ──────────────────────────────────────
            new_best_c   = jnp.where(new_err < best_err, new_c,   best_c)
            new_best_err = jnp.where(new_err < best_err, new_err, best_err)

            # ── Stall detection + random kick ────────────────────────────
            new_stall = jnp.where(improved, jnp.zeros_like(stall_count), stall_count + 1)
            stalled   = new_stall >= _STALL_PATIENCE

            key, subkey = jax.random.split(key)
            kick        = jax.random.normal(subkey, c.shape) * kick_scale
            kick        = jnp.where(fixed_joint_mask, 0.0, kick)  # don't kick fixed joints
            c_out       = jnp.where(stalled, jnp.clip(c_step + kick, lower, upper), c_step)
            lam_out     = jnp.where(stalled, lam0, lam_out)   # reset λ after kick
            stall_out   = jnp.where(stalled, jnp.zeros_like(new_stall), new_stall)

            return (c_out, lam_out, stall_out, key, new_best_c, new_best_err)

        def _no_op(inner):
            return inner

        inner  = (c, lam, stall_count, key, best_c, best_err)
        result = jax.lax.cond(already_done, _no_op, _do_step, inner)
        return (*result, already_done), None

    init_f   = _ik_residual(cfg, robot, target_link_index, target_pose)
    init_err = jnp.dot(init_f, init_f)

    init_carry = (
        cfg,
        lam0,
        jnp.zeros((), dtype=jnp.int32),
        rng_key,
        cfg,                              # best_c  — initialised to starting config
        init_err,                         # best_err — initialised to starting error
        jnp.zeros((), dtype=jnp.bool_),  # done — not yet converged
    )

    (_, _, _, _, best_c, _, _), _ = jax.lax.scan(
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
    static_argnames=("target_link_index", "num_seeds", "coarse_max_iter", "lm_max_iter"),
)
def hjcd_solve(
    robot: Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
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
) -> Float[Array, "n_act"]:
    """Solve IK via the two-phase HJCD-IK algorithm.

    Phase 1 — Coarse search
        Warm seeds near ``previous_cfg`` (σ = 0.05) are mixed with uniform
        random seeds across joint ranges.  All seeds are refined with
        ``coarse_max_iter`` Hamiltonian coordinate-descent steps.

    Phase 2 — LM refinement
        Top-K coarse solutions are tiled ``repeats`` times with small
        Gaussian perturbations (σ = 0.01).  Each copy is refined with the
        advanced LM solver (column scaling, joint-limit prior, line search,
        stall kicks).  Each refinement seed gets an independent RNG key for
        its kick sequence so the batch explores diverse recovery trajectories.

    Winner selection
        The best refined configuration is chosen by task-space residual plus
        ``continuity_weight`` · ‖q − previous_cfg‖².  This weight is used
        **only here** (tie-breaking), never inside the optimisation loop, so
        it cannot cause the "dancing around" instability.

    Args:
        robot:               The robot model.
        target_link_index:   Index of the target link in ``robot.links.names``.
        target_pose:         Desired SE(3) pose for the target link.
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

    warm_keys  = jax.random.split(key_warm, n_warm)
    warm_seeds = jax.vmap(
        lambda k: jnp.clip(
            previous_cfg + jax.random.normal(k, (n_act,)) * 0.05, lower, upper
        )
    )(warm_keys)                                               # (n_warm, n_act)
    warm_seeds = jnp.where(fixed_joint_mask[None, :], previous_cfg[None, :], warm_seeds)

    random_seeds = jax.random.uniform(
        key_seeds, (n_random, n_act), minval=lower, maxval=upper
    )                                                          # (n_random, n_act)
    random_seeds = jnp.where(fixed_joint_mask[None, :], previous_cfg[None, :], random_seeds)

    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=0)  # (B, n_act)

    coarse_cfgs = jax.vmap(
        lambda cfg: _coarse_search_single(
            cfg, robot, target_link_index, target_pose, coarse_max_iter,
            epsilon, nu, lower, upper,
            fixed_joint_mask,
        )
    )(seeds)                                                   # (B, n_act)

    coarse_errors = jax.vmap(
        lambda cfg: jnp.sum(
            _ik_residual(cfg, robot, target_link_index, target_pose) ** 2
        )
    )(coarse_cfgs)                                             # (B,)

    # ── Phase 2: select top-K, perturb, refine with LM ────────────────────
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
            cfg, robot, target_link_index, target_pose,
            lm_max_iter, lambda_init, limit_prior_weight, kick_scale,
            key, lower, upper, eps_pos, eps_ori, fixed_joint_mask,
        )
    )(refine_seeds, lm_keys)                                   # (top_k*repeats, n_act)

    # ── Winner selection: task error + continuity tie-breaker ─────────────
    # continuity_weight acts only here, NOT during optimisation.
    refine_errors = jax.vmap(
        lambda cfg: (
            jnp.sum(_ik_residual(cfg, robot, target_link_index, target_pose) ** 2)
            + continuity_weight * jnp.sum((cfg - previous_cfg) ** 2)
        )
    )(refine_cfgs)

    best_idx = jnp.argmin(refine_errors)
    return refine_cfgs[best_idx]


# ---------------------------------------------------------------------------
# CUDA solver
# ---------------------------------------------------------------------------

def hjcd_solve_cuda(
    robot: Robot,
    target_link_index: int,
    target_pose: jaxlie.SE3,
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
) -> Float[Array, "n_act"]:
    """CUDA alternative to :func:`hjcd_solve`.

    Implements the same two-phase HJCD-IK algorithm but offloads the
    coordinate-descent and Levenberg-Marquardt loops to CUDA kernels via
    the JAX FFI.  The kernels are defined in ``_hjcd_ik_cuda_kernel.cu`` and
    reuse the FK device function from ``_fk_cuda_helpers.cuh``.

    Unlike :func:`hjcd_solve` this function is **not** JIT-compiled by the
    caller — the CUDA kernels are launched eagerly via ``jax.ffi.ffi_call``.
    Seed generation, top-K selection, and winner selection happen in JAX and
    benefit from JAX's own tracing/dispatch cache.

    Key numerical improvement over the JAX path:
      - Normal equations are formed and solved in **float64** inside the CUDA
        kernel, which avoids the ill-conditioning that causes the accuracy
        issues in the float32 JAX path.
      - All kernel launches use the caller's CUDA stream so there are no
        implicit device synchronisations.

    Requires ``_hjcd_ik_cuda_lib.so`` to be compiled first:
        bash src/pyronot/cuda_kernels/build_hjcd_ik_cuda.sh

    Args:
        robot:               The robot model.
        target_link_index:   Index into ``robot.links.names``.
        target_pose:         Desired SE(3) world pose for the target link.
        rng_key:             JAX PRNG key.
        previous_cfg:        Previous joint configuration (warm-start + continuity).
        num_seeds:           Coarse-phase batch size B.
        coarse_max_iter:     CD iteration budget.
        lm_max_iter:         LM iterations per refinement seed.
        epsilon:             Position convergence threshold [m] used by
                             Hamiltonian coarse mode early-stop.
        nu:                  Orientation convergence threshold [rad] used by
                             Hamiltonian coarse mode early-stop.
        eps_pos:             Position convergence tolerance [m] for LM early exit.
        eps_ori:             Orientation convergence tolerance [rad] for LM early exit.
        lambda_init:         Initial LM damping factor.
        continuity_weight:   Weight on ‖q − previous_cfg‖² in winner selection.
        limit_prior_weight:  Strength of soft joint-limit prior in LM.
        kick_scale:          Std-dev of random kicks in LM stall recovery.
        fixed_joint_mask:    Boolean mask; True = joint must not move.

    Returns:
        Best joint configuration found, shape ``(n_act,)``.
    """
    from ..cuda_kernels._hjcd_ik_cuda import hjcd_ik_coarse_cuda, hjcd_ik_lm_cuda

    n_act  = robot.joints.num_actuated_joints
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    if fixed_joint_mask is None:
        fixed_joint_mask_int = jnp.zeros(n_act, dtype=jnp.int32)
    else:
        fixed_joint_mask_int = fixed_joint_mask.astype(jnp.int32)
    # ── Pre-compute Python-level values (need concrete arrays) ────────────
    # These cannot be computed inside jax.jit because parent_joint_indices
    # and parent_indices are JAX-array leaves of the Robot pytree.
    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    target_joint_idx = int(parent_joint_indices_np[target_link_index])

    parent_idx_np = np.array(robot.joints.parent_indices)
    ancestor_mask_np = np.zeros(robot.joints.num_joints, dtype=np.int32)
    j = target_joint_idx
    while j >= 0:
        ancestor_mask_np[j] = 1
        j = int(parent_idx_np[j])
    ancestor_mask = jnp.array(ancestor_mask_np)

    # Target pose as [w, x, y, z, tx, ty, tz] float32.
    target_T = target_pose.wxyz_xyz.astype(jnp.float32)

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

    # ── Phase 1: CUDA coarse coordinate descent ───────────────────────────
    coarse_cfgs, coarse_errors = hjcd_ik_coarse_cuda(
        seeds=seeds,
        twists=robot.joints.twists,
        parent_tf=robot.joints.parent_transforms,
        parent_idx=robot.joints.parent_indices,
        act_idx=robot.joints.actuated_indices,
        mimic_mul=robot.joints.mimic_multiplier,
        mimic_off=robot.joints.mimic_offset,
        mimic_act_idx=robot.joints.mimic_act_indices,
        topo_inv=robot.joints._topo_sort_inv,
        ancestor_mask=ancestor_mask,
        target_T=target_T,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_joint_mask_int,
        target_jnt=target_joint_idx,
        k_max=coarse_max_iter,
    )  # coarse_cfgs: (num_seeds, n_act), coarse_errors: (num_seeds,)

    # ── Phase 2 setup: top-K selection + perturbation ─────────────────────
    # Errors returned directly by the CUDA kernel (no Python-side FK rescore).
    # jax.lax.top_k maps to the XLA TopK op — stays on-device with no
    # GPU→CPU→GPU round-trip that jnp.argsort would force via materialisation.
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

    # ── Phase 2: CUDA Levenberg-Marquardt refinement ──────────────────────
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
        ancestor_mask=ancestor_mask,
        target_T=target_T,
        lower=lower,
        upper=upper,
        fixed_mask=fixed_joint_mask_int,
        target_jnt=target_joint_idx,
        max_iter=lm_max_iter,
        stall_patience=_STALL_PATIENCE,
        lambda_init=float(lambda_init),
        limit_prior_weight=float(limit_prior_weight),
        kick_scale=float(kick_scale),
        eps_pos=float(eps_pos),
        eps_ori=float(eps_ori),
    )  # refine_cfgs: (n_lm_seeds, n_act), refine_errors_raw: (n_lm_seeds,)

    # ── Winner selection: task error + continuity tie-breaker ─────────────
    # Errors returned directly by the CUDA kernel (no Python-side FK rescore).
    refine_errors = (
        refine_errors_raw
        + continuity_weight * jnp.sum((refine_cfgs - previous_cfg) ** 2, axis=-1)
    )

    best_idx = jnp.argmin(refine_errors)
    return refine_cfgs[best_idx]
