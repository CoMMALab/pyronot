"""MPPI + L-BFGS IK Solver.

Two-stage inverse kinematics solver following the cuRobo approach:

  Stage 1 — MPPI coarse stochastic search
    Model Predictive Path Integral (MPPI) particle optimisation.  At each
    iteration K Gaussian-noise perturbations are sampled around the current
    seed, forward kinematics and the SE(3) cost are evaluated for every
    particle, and a temperature-weighted mean update moves the seed toward
    the low-cost region of configuration space.  No Jacobians are required.
    The best config seen across all MPPI iterations is retained as the
    warm-start for Stage 2.

  Stage 2 — L-BFGS gradient refinement
    Limited-memory BFGS (Nocedal two-loop, m history pairs) starting from
    the MPPI warm-start.  Gradient: g = J_w^T (W r).  5-point line search
    with sufficient-decrease early exit and adaptive trust-region step
    clipping identical to the CUDA kernel.  A convergence gate skips Stage 2
    entirely if MPPI already satisfies eps_pos / eps_ori for all EEs.

CUDA kernel
    Offloads both stages to a single CUDA kernel via JAX FFI.  The kernel
    uses a per-thread xorshift32 PRNG seeded from the JAX PRNG key so
    results are deterministic given the same key.

JAX reference implementation
    ``mppi_ik_solve`` provides a pure-JAX variant for debugging and
    non-CUDA environments.  The MPPI inner loop uses ``jax.vmap`` over
    particles and tracks the best config across iterations.  The L-BFGS
    stage uses Nocedal two-loop with 5-point line search, trust-region
    step clipping, and a convergence gate matching the CUDA kernel.

Shared with LS-IK / SQP-IK
  - Same seed generation (warm + random, multi-EE tight/loose split).
  - Same continuity-weighted winner selection.
  - Same CUDA grid/block layout.
  - Same per-EE ancestor mask pre-computation.
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
from ._ik_primitives import _ik_residual, _LS_ALPHAS, split_cuda_and_post_constraints
from ._ls_ik import _ls_ik_single, _prepare_ls_collision_buffers


# ---------------------------------------------------------------------------
# L-BFGS two-loop helper (Nocedal, mirrors CUDA lbfgs_two_loop)
# ---------------------------------------------------------------------------

def _lbfgs_two_loop(
    g:       Float[Array, "n_act"],
    s_buf:   Float[Array, "m n_act"],
    y_buf:   Float[Array, "m n_act"],
    rho_buf: Float[Array, "m"],
    m_used:  Array,   # traced int32
    newest:  Array,   # traced int32
    m_lbfgs: int,     # static Python int → unrolled
) -> Float[Array, "n_act"]:
    """Nocedal two-loop recursion: returns -H*g.

    m_lbfgs is a static Python int so both loops are unrolled at trace time.
    Active entries are gated by ``step < m_used``; inactive steps are no-ops.
    When m_used == 0 the function returns the zero vector; the caller must
    fall back to the normalized gradient direction.
    """
    # alpha_arr is workspace indexed by circular buffer position.
    alpha_arr = jnp.zeros(m_lbfgs)
    q = g

    # Forward pass: newest → oldest
    for i in range(m_lbfgs):
        buf_idx = (newest - i + m_lbfgs) % m_lbfgs
        active  = i < m_used
        si      = s_buf[buf_idx]
        yi      = y_buf[buf_idx]
        rho_i   = rho_buf[buf_idx]
        alpha_i = rho_i * jnp.dot(si, q)
        alpha_arr = jnp.where(active, alpha_arr.at[buf_idx].set(alpha_i), alpha_arr)
        q = jnp.where(active, q - alpha_i * yi, q)

    # Shanno-Kettler H₀ scaling using the most-recent pair.
    sy    = jnp.dot(s_buf[newest], y_buf[newest])
    yy    = jnp.dot(y_buf[newest], y_buf[newest])
    gamma = sy / (yy + 1e-18)
    r     = gamma * q

    # Backward pass: oldest → newest
    for step in range(m_lbfgs):
        buf_idx = (newest - m_used + 1 + step + m_lbfgs) % m_lbfgs
        active  = step < m_used
        si      = s_buf[buf_idx]
        yi      = y_buf[buf_idx]
        rho_i   = rho_buf[buf_idx]
        alpha_i = alpha_arr[buf_idx]
        beta    = rho_i * jnp.dot(yi, r)
        r       = jnp.where(active, r + si * (alpha_i - beta), r)

    return -r


# ---------------------------------------------------------------------------
# True L-BFGS single-seed refinement (mirrors CUDA Stage 2)
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "target_link_indices", "n_lbfgs_iters", "m_lbfgs",
        "eps_pos", "eps_ori", "constraint_fns",
    ),
)
def _lbfgs_ik_single(
    cfg:                  Float[Array, "n_act"],
    robot:                Robot,
    target_link_indices:  tuple[int, ...],
    target_poses:         tuple,
    n_lbfgs_iters:        int,
    m_lbfgs:              int,
    pos_weight:           float,
    ori_weight:           float,
    lower:                Float[Array, "n_act"],
    upper:                Float[Array, "n_act"],
    fixed_joint_mask:     Float[Array, "n_act"],
    eps_pos:              float = 1e-8,
    eps_ori:              float = 1e-8,
    constraint_fns:       tuple = (),
    constraint_args:      tuple = (),
    constraint_weights:   Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    """True Nocedal two-loop L-BFGS IK refinement.

    Mirrors CUDA Stage 2 exactly (except JAX-native vmap line search):
      - Gradient: g = J_w^T (W r)  via jax.value_and_grad
      - History update with positive-curvature guard  (s^T y > ε)
      - Shanno-Kettler H₀ scaling
      - Trust-region step clipping (4 adaptive radii matching CUDA)
      - 5-point line search; first sufficient-decrease alpha wins
      - Per-iteration convergence check (eps_pos, eps_ori for each EE)
      - Best-config tracking across all iterations
    """
    n_act = cfg.shape[0]
    n_ee  = len(target_link_indices)
    n_c   = len(constraint_fns)
    W_ee  = jnp.concatenate([jnp.full(3, pos_weight), jnp.full(3, ori_weight)])

    if n_c > 0:
        sqrt_wc = jnp.sqrt(constraint_weights)

    def cost_half_and_residuals(q):
        """Returns (half_cost, residuals) for combined value+grad+aux in one pass."""
        residuals = jnp.stack([
            _ik_residual(q, robot, target_link_indices[i], target_poses[i])
            for i in range(n_ee)
        ])  # (n_ee, 6)
        total = jnp.sum((residuals * W_ee[None, :]) ** 2) * 0.5
        if n_c > 0:
            f_coll = jnp.stack([
                constraint_fns[i](q, robot, constraint_args[i])
                for i in range(n_c)
            ])
            total = total + jnp.dot(sqrt_wc * f_coll, sqrt_wc * f_coll) * 0.5
        return total, residuals

    def cost_half(q):
        return cost_half_and_residuals(q)[0]

    # Initial gradient.
    (init_cost, _), init_g = jax.value_and_grad(
        cost_half_and_residuals, has_aux=True
    )(cfg)
    init_g = jnp.where(fixed_joint_mask, 0.0, init_g)

    # ── lax.scan carry ─────────────────────────────────────────────────────
    # (cfg, best_cfg, best_cost, cfg_prev, g_prev,
    #  s_buf, y_buf, rho_buf, m_used, newest, converged, iter_count)
    init_carry = (
        cfg,
        cfg,
        init_cost,
        cfg,                            # cfg_prev — dummy for iter 0
        jnp.zeros(n_act),               # g_prev   — dummy for iter 0
        jnp.zeros((m_lbfgs, n_act)),
        jnp.zeros((m_lbfgs, n_act)),
        jnp.zeros(m_lbfgs),
        jnp.int32(0),                   # m_used
        jnp.int32(0),                   # newest   — valid after first update
        jnp.bool_(False),               # converged
        jnp.int32(0),                   # iter_count
    )

    def lbfgs_body(carry, _):
        (cfg_c, best_cfg, best_cost, cfg_prev, g_prev,
         s_buf, y_buf, rho_buf, m_used, newest, converged, iter_count) = carry

        # ── Gradient at current config ────────────────────────────────────
        (half_cost, residuals), g = jax.value_and_grad(
            cost_half_and_residuals, has_aux=True
        )(cfg_c)
        g = jnp.where(fixed_joint_mask, 0.0, g)

        # ── Per-iteration convergence check ───────────────────────────────
        pos_norms = jnp.linalg.norm(residuals[:, :3], axis=-1)  # (n_ee,)
        ori_norms = jnp.linalg.norm(residuals[:, 3:], axis=-1)  # (n_ee,)
        converged = converged | (
            jnp.all(pos_norms < eps_pos) & jnp.all(ori_norms < eps_ori)
        )

        # ── Update L-BFGS history ─────────────────────────────────────────
        # s_k = Δcfg,  y_k = Δgrad.  Skip on iter 0 (no valid prev yet).
        s_k   = cfg_c - cfg_prev
        y_k   = g     - g_prev
        sy    = jnp.dot(s_k, y_k)
        yy    = jnp.dot(y_k, y_k)
        valid = (sy > 1e-10 * yy + 1e-30) & (iter_count > 0)

        new_newest     = (newest + 1) % m_lbfgs
        actual_newest  = jnp.where(valid, new_newest, newest)
        s_buf   = jnp.where(valid, s_buf.at[new_newest].set(s_k),       s_buf)
        y_buf   = jnp.where(valid, y_buf.at[new_newest].set(y_k),       y_buf)
        rho_buf = jnp.where(valid, rho_buf.at[new_newest].set(1.0 / sy), rho_buf)
        m_used  = jnp.where(valid & (m_used < m_lbfgs), m_used + 1,     m_used)
        newest  = actual_newest

        # ── L-BFGS two-loop direction ─────────────────────────────────────
        dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m_lbfgs)
        dir_gd    = -g / (jnp.linalg.norm(g) + 1e-18)
        direction = jnp.where(m_used > 0, dir_lbfgs, dir_gd)
        direction = jnp.where(fixed_joint_mask, 0.0, direction)

        # ── Trust-region step clipping (4 adaptive radii, same as CUDA) ──
        max_p = jnp.max(pos_norms)
        max_o = jnp.max(ori_norms)
        R = jnp.where(
            (max_p > 1e-2) | (max_o > 0.6),   0.38,
            jnp.where(
            (max_p > 1e-3) | (max_o > 0.25),  0.22,
            jnp.where(
            (max_p > 2e-4) | (max_o > 0.08),  0.12,
            0.05)))
        dnorm     = jnp.linalg.norm(direction)
        direction = direction * jnp.where(dnorm > R, R / (dnorm + 1e-18), 1.0)

        # ── 5-point line search (vmapped; first sufficient-decrease wins) ─
        # Mirrors CUDA: first alpha satisfying err < curr*(1-1e-4) is chosen;
        # fall back to argmin if none qualifies.
        suff_thresh = half_cost * (1.0 - 1e-4)

        def eval_alpha(alpha):
            return cost_half(jnp.clip(cfg_c + alpha * direction, lower, upper))

        trial_costs  = jax.vmap(eval_alpha)(_LS_ALPHAS)   # (5,)
        has_suff     = trial_costs < suff_thresh
        best_ls_idx  = jnp.where(
            jnp.any(has_suff),
            jnp.argmax(has_suff),        # first sufficient-decrease alpha
            jnp.argmin(trial_costs),     # fall back to best
        )
        best_alpha   = _LS_ALPHAS[best_ls_idx]
        step_cost    = trial_costs[best_ls_idx]
        cfg_new      = jnp.clip(cfg_c + best_alpha * direction, lower, upper)

        # ── Best-config tracking ──────────────────────────────────────────
        improved  = step_cost < best_cost
        best_cfg  = jnp.where(improved, cfg_new, best_cfg)
        best_cost = jnp.where(improved, step_cost, best_cost)

        # ── Apply step only if not converged ──────────────────────────────
        cfg_out = jnp.where(converged, cfg_c, cfg_new)

        new_carry = (
            cfg_out, best_cfg, best_cost,
            cfg_c, g,          # save current (cfg, grad) as prev for next iter
            s_buf, y_buf, rho_buf, m_used, newest,
            converged, iter_count + 1,
        )
        return new_carry, None

    (_, best_cfg, _, _, _, _, _, _, _, _, _, _), _ = jax.lax.scan(
        lbfgs_body, init_carry, None, length=n_lbfgs_iters,
    )
    return best_cfg


# ---------------------------------------------------------------------------
# JAX single-seed MPPI step helper
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("target_link_indices", "n_particles", "constraint_fns"),
)
def _mppi_step(
    cfg:                  Float[Array, "n_act"],
    rng_key:              Array,
    robot:                Robot,
    target_link_indices:  tuple[int, ...],
    target_poses:         tuple,
    pos_weight:           float,
    ori_weight:           float,
    sigma:                float,
    mppi_temperature:     float,
    lower:                Float[Array, "n_act"],
    upper:                Float[Array, "n_act"],
    fixed_joint_mask:     Float[Array, "n_act"],
    n_particles:          int,
    constraint_fns:       tuple = (),
    constraint_args:      tuple = (),
    constraint_weights:   Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    """Single MPPI update step.

    Samples n_particles perturbations, evaluates the weighted SE(3) cost for
    each, computes MPPI weights, and returns the weighted-mean-updated cfg.
    """
    n_act = cfg.shape[0]
    n_ee  = len(target_link_indices)
    n_c   = len(constraint_fns)
    W = jnp.concatenate([
        jnp.full(3, pos_weight, dtype=cfg.dtype),
        jnp.full(3, ori_weight, dtype=cfg.dtype),
    ])

    if n_c > 0:
        sqrt_wc = jnp.sqrt(constraint_weights)

    def cost_fn(q: Array) -> Array:
        parts: list[Array] = []
        for i in range(n_ee):
            f_i = _ik_residual(q, robot, target_link_indices[i], target_poses[i])
            parts.append(f_i * W)
        if n_c > 0:
            f_coll = jnp.stack([
                constraint_fns[i](q, robot, constraint_args[i])
                for i in range(n_c)
            ])
            parts.append(sqrt_wc * f_coll)
        f = jnp.concatenate(parts)
        return jnp.dot(f, f)

    noise = jax.random.normal(rng_key, (n_particles, n_act)) * sigma
    noise = jnp.where(fixed_joint_mask[None, :], 0.0, noise)

    def eval_particle(noise_k: Array) -> Array:
        q_trial = jnp.clip(cfg + noise_k, lower, upper)
        return cost_fn(q_trial)

    costs   = jax.vmap(eval_particle)(noise)   # (K,)
    min_c   = jnp.min(costs)
    weights = jnp.exp(-(costs - min_c) / jnp.maximum(mppi_temperature, 1e-8))
    weights = weights / (jnp.sum(weights) + 1e-20)

    delta   = jnp.sum(weights[:, None] * noise, axis=0)
    return jnp.clip(cfg + delta, lower, upper)


# ---------------------------------------------------------------------------
# Public entry point — JAX
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "target_link_indices", "num_seeds", "n_particles",
        "n_mppi_iters", "n_lbfgs_iters", "m_lbfgs",
        "eps_pos", "eps_ori", "constraint_fns",
    ),
)
def mppi_ik_solve(
    robot:               Robot,
    target_link_indices: tuple[int, ...],
    target_poses:        tuple,
    rng_key:             Array,
    previous_cfg:        Float[Array, "n_act"],
    num_seeds:           int   = 32,
    n_particles:         int   = 16,
    n_mppi_iters:        int   = 5,
    n_lbfgs_iters:       int   = 30,
    m_lbfgs:             int   = 5,
    pos_weight:          float = 50.0,
    ori_weight:          float = 10.0,
    sigma:               float = 0.3,
    mppi_temperature:    float = 0.05,
    eps_pos:             float = 1e-8,
    eps_ori:             float = 1e-8,
    continuity_weight:   float = 0.0,
    fixed_joint_mask:    Float[Array, "n_act"] | None = None,
    constraint_fns:      tuple = (),
    constraint_args:     tuple = (),
    constraint_weights:  Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    """Solve IK via MPPI coarse search followed by true L-BFGS refinement.

    Aligned with the CUDA kernel:
      Stage 1 (MPPI): ``jax.vmap`` over particles; best config across all
        iterations is retained as the L-BFGS warm-start.
      Stage 2 (L-BFGS): Nocedal two-loop with 5-point line search, adaptive
        trust-region clipping, and per-iteration convergence checks.
      Convergence gate: L-BFGS is skipped if MPPI already satisfies
        ``eps_pos`` / ``eps_ori`` for every EE.

    JAX-specific: vmap over particles (vs. sequential in CUDA), and
    constraint residuals are embedded directly in the cost (vs. post-kernel
    in the CUDA path).

    Args:
        robot:               The robot model.
        target_link_indices: Tuple of target link indices (static for JIT).
        target_poses:        Tuple of desired SE(3) world poses.
        rng_key:             JAX PRNG key.
        previous_cfg:        Previous joint configuration for warm-starting.
        num_seeds:           Number of parallel seeds.
        n_particles:         MPPI particles per step.
        n_mppi_iters:        MPPI stage iterations.
        n_lbfgs_iters:       L-BFGS stage iterations.
        m_lbfgs:             L-BFGS history size.
        pos_weight:          Weight on position residual.
        ori_weight:          Weight on orientation residual.
        sigma:               MPPI noise std dev [rad/m].
        mppi_temperature:    MPPI softmax temperature.
        eps_pos:             Position convergence threshold [m].
        eps_ori:             Orientation convergence threshold [rad].
        continuity_weight:   Weight on ‖q − prev‖² in winner selection.
        fixed_joint_mask:    Boolean mask; True = joint must not move.
        constraint_fns:      Tuple of constraint callables.
        constraint_weights:  Weight per constraint.

    Returns:
        Best joint configuration found, shape ``(n_actuated_joints,)``.
    """
    n_act  = robot.joints.num_actuated_joints
    n_ee   = len(target_link_indices)
    n_c    = len(constraint_fns)
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    if fixed_joint_mask is None:
        fixed_joint_mask = jnp.zeros(n_act, dtype=jnp.bool_)

    W_ee = jnp.concatenate([jnp.full(3, pos_weight), jnp.full(3, ori_weight)])
    if n_c > 0:
        sqrt_wc = jnp.sqrt(constraint_weights)

    # Cost for MPPI best-tracking (full squared weighted residual).
    def pose_cost(q: Array) -> Array:
        total = jnp.zeros(())
        for i in range(n_ee):
            r_i = _ik_residual(q, robot, target_link_indices[i], target_poses[i])
            wr  = r_i * W_ee
            total = total + jnp.dot(wr, wr)
        if n_c > 0:
            f_coll = jnp.stack([
                constraint_fns[i](q, robot, constraint_args[i])
                for i in range(n_c)
            ])
            total = total + jnp.dot(sqrt_wc * f_coll, sqrt_wc * f_coll)
        return total

    # ── Seed generation (identical to LS/SQP) ─────────────────────────────
    n_warm   = max(1, num_seeds // 2)
    n_random = num_seeds - n_warm

    key_warm, key_random, key_mppi = jax.random.split(rng_key, 3)

    if len(target_link_indices) > 1:
        n_tight = max(1, n_warm // 2)
        n_loose = n_warm - n_tight
        key_tight, key_loose = jax.random.split(key_warm)
        tight_seeds = jnp.clip(
            previous_cfg[None, :] + jax.random.normal(key_tight, (n_tight, n_act)) * 0.05,
            lower, upper,
        )
        loose_seeds = jnp.clip(
            previous_cfg[None, :] + jax.random.normal(key_loose, (n_loose, n_act)) * 0.3,
            lower, upper,
        )
        warm_seeds = jnp.concatenate([tight_seeds, loose_seeds], axis=0)
    else:
        warm_seeds = jnp.clip(
            previous_cfg[None, :] + jax.random.normal(key_warm, (n_warm, n_act)) * 0.05,
            lower, upper,
        )
    warm_seeds = jnp.where(fixed_joint_mask[None, :], previous_cfg[None, :], warm_seeds)

    random_seeds = jax.random.uniform(
        key_random, (n_random, n_act), minval=lower, maxval=upper
    )
    random_seeds = jnp.where(
        fixed_joint_mask[None, :], previous_cfg[None, :], random_seeds
    )
    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=0)  # (num_seeds, n_act)

    # ── Stage 1: MPPI coarse search with best-cfg tracking ────────────────
    def mppi_refine(seed: Array, seed_key: Array) -> Array:
        init_cost = pose_cost(seed)

        def mppi_scan_step(carry, _):
            cfg, best_cfg, best_cost, key = carry
            key, sub = jax.random.split(key)
            cfg = _mppi_step(
                cfg, sub, robot, target_link_indices, target_poses,
                pos_weight, ori_weight, sigma, mppi_temperature,
                lower, upper, fixed_joint_mask, n_particles,
                constraint_fns=constraint_fns,
                constraint_args=constraint_args,
                constraint_weights=constraint_weights,
            )
            curr_cost = pose_cost(cfg)
            improved  = curr_cost < best_cost
            return (
                cfg,
                jnp.where(improved, cfg,       best_cfg),
                jnp.where(improved, curr_cost, best_cost),
                key,
            ), None

        (_, best_cfg, _, _), _ = jax.lax.scan(
            mppi_scan_step, (seed, seed, init_cost, seed_key), None, length=n_mppi_iters,
        )
        return best_cfg

    seed_keys = jax.random.split(key_mppi, num_seeds)
    mppi_cfgs = jax.vmap(mppi_refine)(seeds, seed_keys)  # (num_seeds, n_act)

    # ── Stage 2: convergence gate + true L-BFGS refinement ────────────────
    def refine_with_gate(cfg: Array) -> Array:
        residuals = jnp.stack([
            _ik_residual(cfg, robot, target_link_indices[i], target_poses[i])
            for i in range(n_ee)
        ])  # (n_ee, 6)
        all_conv = (
            jnp.all(jnp.linalg.norm(residuals[:, :3], axis=-1) < eps_pos) &
            jnp.all(jnp.linalg.norm(residuals[:, 3:], axis=-1) < eps_ori)
        )
        return jax.lax.cond(
            all_conv,
            lambda: cfg,
            lambda: _lbfgs_ik_single(
                cfg, robot, target_link_indices, target_poses,
                n_lbfgs_iters, m_lbfgs, pos_weight, ori_weight,
                lower, upper, fixed_joint_mask, eps_pos, eps_ori,
                constraint_fns=constraint_fns,
                constraint_args=constraint_args,
                constraint_weights=constraint_weights,
            ),
        )

    all_cfgs = jax.vmap(refine_with_gate)(mppi_cfgs)  # (num_seeds, n_act)

    # ── Winner selection ──────────────────────────────────────────────────
    W = jnp.concatenate([jnp.full(3, pos_weight), jnp.full(3, ori_weight)])

    def weighted_err(cfg: Float[Array, "n_act"]) -> Array:
        err = sum(
            jnp.sum((_ik_residual(cfg, robot, target_link_indices[i], target_poses[i]) * W) ** 2)
            for i in range(n_ee)
        )
        if n_c > 0:
            c_vals = jnp.stack([
                constraint_fns[i](cfg, robot, constraint_args[i])
                for i in range(n_c)
            ])
            err = err + jnp.sum(constraint_weights * c_vals ** 2)
        return err + continuity_weight * jnp.sum((cfg - previous_cfg) ** 2)

    errors   = jax.vmap(weighted_err)(all_cfgs)
    best_idx = jnp.argmin(errors)
    return all_cfgs[best_idx]


# ---------------------------------------------------------------------------
# Public entry point — CUDA (JIT inner)
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "n_particles",
        "n_mppi_iters",
        "n_lbfgs_iters",
        "m_lbfgs",
        "pos_weight",
        "ori_weight",
        "sigma",
        "mppi_temperature",
        "eps_pos",
        "eps_ori",
        "enable_collision",
        "collision_weight",
        "collision_margin",
        "constraint_fns",
        "target_link_indices",
    ),
)
def _mppi_ik_solve_cuda_jit(
    robot:                Robot,
    target_poses:         tuple,
    rng_key:              Array,
    previous_cfg:         Float[Array, "n_act"],
    num_seeds:            int,
    n_particles:          int,
    n_mppi_iters:         int,
    n_lbfgs_iters:        int,
    m_lbfgs:              int,
    pos_weight:           float,
    ori_weight:           float,
    sigma:                float,
    mppi_temperature:     float,
    eps_pos:              float,
    eps_ori:              float,
    continuity_weight:    float,
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
) -> tuple[Float[Array, "n_act"], Array]:
    from ..cuda_kernels._mppi_ik_cuda import mppi_ik_cuda

    n_act  = robot.joints.num_actuated_joints
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    target_Ts = jnp.stack([tp.wxyz_xyz.astype(jnp.float32) for tp in target_poses], axis=0)

    # ── Seed generation ────────────────────────────────────────────────────
    n_warm   = max(1, num_seeds // 2)
    n_random = num_seeds - n_warm

    key_warm, key_random, key_seed = jax.random.split(rng_key, 3)

    warm_seeds = jnp.clip(
        previous_cfg[None, :] + jax.random.normal(key_warm, (n_warm, n_act)) * 0.05,
        lower, upper,
    )
    warm_seeds = jnp.where(fixed_joint_mask_int[None, :], previous_cfg[None, :], warm_seeds)

    random_seeds = jax.random.uniform(
        key_random, (n_random, n_act), minval=lower, maxval=upper
    )
    random_seeds = jnp.where(
        fixed_joint_mask_int[None, :], previous_cfg[None, :], random_seeds
    )
    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=0)  # (num_seeds, n_act)

    # Scalar int32 tensor used as on-device RNG seed — traced so it varies
    # each call without triggering JIT recompilation.
    rng_seed_arr = jax.random.bits(key_seed, dtype=jnp.uint32).astype(jnp.int32)

    cfgs, errors = mppi_ik_cuda(
        seeds          = seeds[None],           # (1, n_seeds, n_act)
        twists         = robot.joints.twists,
        parent_tf      = robot.joints.parent_transforms,
        parent_idx     = robot.joints.parent_indices,
        act_idx        = robot.joints.actuated_indices,
        mimic_mul      = robot.joints.mimic_multiplier,
        mimic_off      = robot.joints.mimic_offset,
        mimic_act_idx  = robot.joints.mimic_act_indices,
        topo_inv       = robot.joints._topo_sort_inv,
        target_jnts    = target_jnts,
        ancestor_masks = ancestor_masks,
        target_T       = target_Ts[None],       # (1, n_ee, 7)
        robot_spheres_local = robot_spheres_local,
        robot_sphere_joint_idx = robot_sphere_joint_idx,
        world_spheres = world_spheres,
        world_capsules = world_capsules,
        world_boxes = world_boxes,
        world_halfspaces = world_halfspaces,
        lower          = lower,
        upper          = upper,
        fixed_mask     = fixed_joint_mask_int,
        rng_seed       = rng_seed_arr,
        n_particles      = n_particles,
        n_mppi_iters     = n_mppi_iters,
        n_lbfgs_iters    = n_lbfgs_iters,
        m_lbfgs          = m_lbfgs,
        sigma            = sigma,
        mppi_temperature = mppi_temperature,
        pos_weight       = pos_weight,
        ori_weight       = ori_weight,
        eps_pos          = eps_pos,
        eps_ori          = eps_ori,
        enable_collision = enable_collision,
        collision_weight = collision_weight,
        collision_margin = collision_margin,
    )
    cfgs   = cfgs[0]    # (n_seeds, n_act)
    errors = errors[0]  # (n_seeds,)

    if len(constraint_fns) > 0:
        def constraint_penalty(cfg):
            c_vals = jnp.stack([
                constraint_fns[i](cfg, robot, constraint_args[i])
                for i in range(len(constraint_fns))
            ])
            return jnp.sum(constraint_weights * c_vals ** 2)

        constraint_errors = jax.vmap(constraint_penalty)(cfgs)
        final_errors = (
            errors
            + constraint_errors
            + continuity_weight * jnp.sum((cfgs - previous_cfg) ** 2, axis=-1)
        )
    else:
        constraint_errors = jnp.zeros(cfgs.shape[0])
        final_errors = (
            errors
            + continuity_weight * jnp.sum((cfgs - previous_cfg) ** 2, axis=-1)
        )

    best_idx = jnp.argmin(final_errors)
    if len(constraint_fns) > 0:
        return cfgs[best_idx], constraint_errors[best_idx]
    return cfgs[best_idx], jnp.zeros(())


def mppi_ik_solve_cuda(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3 | tuple,
    rng_key:             Array,
    previous_cfg:        Float[Array, "n_act"],
    num_seeds:           int   = 32,
    n_particles:         int   = 16,
    n_mppi_iters:        int   = 5,
    n_lbfgs_iters:       int   = 25,
    m_lbfgs:             int   = 5,
    pos_weight:          float = 50.0,
    ori_weight:          float = 10.0,
    sigma:               float = 0.3,
    mppi_temperature:    float = 0.05,
    eps_pos:             float = 1e-8,
    eps_ori:             float = 1e-8,
    continuity_weight:   float = 0.0,
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
    constraint_refine_iters: int = 12,
) -> Float[Array, "n_act"]:
    """CUDA MPPI+L-BFGS IK: coarse particle search then gradient refinement.

    Requires ``_mppi_ik_cuda_lib.so`` compiled from ``_mppi_ik_cuda_kernel.cu``:
        bash src/pyroffi/cuda_kernels/build_mppi_ik_cuda.sh

    Stage 1 (MPPI)
        At each of ``n_mppi_iters`` iterations, ``n_particles`` Gaussian
        perturbations are sampled on-device (xorshift32 PRNG), evaluated
        (FK + SE(3) cost, no Jacobian), and a temperature-weighted mean
        update is applied to the seed.  The on-device RNG is seeded from
        a 32-bit integer derived from ``rng_key`` mixed with the thread and
        problem indices, ensuring deterministic but diverse noise.

    Stage 2 (L-BFGS)
        ``n_lbfgs_iters`` Nocedal two-loop L-BFGS steps, each followed by a
        5-point line search and trust-region clipping.  The ``m_lbfgs``
        most recent (s, y) history pairs are used.  New pairs are only
        stored when the positive-curvature condition s^T y > 0 holds.

    Args:
        robot:                   The robot model.
        target_link_indices:     Index (or tuple of indices) of target link(s).
        target_poses:            Desired SE(3) world pose (or tuple of poses).
        rng_key:                 JAX PRNG key (used for seed generation + device RNG seed).
        previous_cfg:            Previous joint configuration.
        num_seeds:               Number of parallel seeds.
        n_particles:             MPPI particles per step (max 32 in default build).
        n_mppi_iters:            MPPI stage iterations.
        n_lbfgs_iters:           L-BFGS stage iterations.
        m_lbfgs:                 L-BFGS history size (max 8 in default build).
        pos_weight:              Weight on position residual.
        ori_weight:              Weight on orientation residual.
        sigma:                   MPPI noise std dev [rad/m].
        mppi_temperature:        MPPI softmax temperature.
        eps_pos:                 Position convergence threshold [m].
        eps_ori:                 Orientation convergence threshold [rad].
        continuity_weight:       Weight on ‖q − prev‖² in winner selection.
        fixed_joint_mask:        Boolean mask; True = joint must not move.
        constraints:             List of constraint callables.
        constraint_weights:      Scalar weight per constraint.
        constraint_refine_iters: Post-CUDA JAX LM refinement on the winner.

    Returns:
        Best joint configuration found, shape ``(n_act,)``.
    """
    if isinstance(target_link_indices, int):
        target_link_indices = (target_link_indices,)
    if isinstance(target_poses, jaxlie.SE3):
        target_poses = (target_poses,)
    target_poses_t = tuple(target_poses)

    n_act = robot.joints.num_actuated_joints

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

    # ── Pre-compute per-EE ancestor masks ──────────────────────────────────
    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    parent_idx_np           = np.array(robot.joints.parent_indices)
    n_joints                = robot.joints.num_joints
    n_ee_count              = len(target_link_indices)

    target_joints_np  = np.zeros(n_ee_count, dtype=np.int32)
    ancestor_masks_np = np.zeros((n_ee_count, n_joints), dtype=np.int32)
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

    winner, winner_coll_cost = _mppi_ik_solve_cuda_jit(
        robot=robot,
        target_poses=target_poses_t,
        rng_key=rng_key,
        previous_cfg=previous_cfg,
        num_seeds=num_seeds,
        n_particles=n_particles,
        n_mppi_iters=n_mppi_iters,
        n_lbfgs_iters=n_lbfgs_iters,
        m_lbfgs=m_lbfgs,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        sigma=sigma,
        mppi_temperature=mppi_temperature,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        continuity_weight=continuity_weight,
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

    # ── Optional post-CUDA JAX constraint refinement ─────────────────────
    if bool(post_constraint_fns) and constraint_refine_iters > 0:
        fmask = (
            fixed_joint_mask.astype(jnp.bool_)
            if fixed_joint_mask is not None
            else jnp.zeros(n_act, dtype=jnp.bool_)
        )
        winner = _ls_ik_single(
            winner, robot, target_link_indices, target_poses_t,
            constraint_refine_iters, 5e-3, pos_weight, ori_weight,
            robot.joints.lower_limits, robot.joints.upper_limits,
            fmask,
            constraint_fns=post_constraint_fns,
            constraint_args=post_constraint_args,
            constraint_weights=post_constraint_weights,
        )

    return winner


# ---------------------------------------------------------------------------
# Public entry point — CUDA batched
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "n_particles",
        "n_mppi_iters",
        "n_lbfgs_iters",
        "m_lbfgs",
        "pos_weight",
        "ori_weight",
        "sigma",
        "mppi_temperature",
        "eps_pos",
        "eps_ori",
        "enable_collision",
        "collision_weight",
        "collision_margin",
        "constraint_fns",
        "target_link_indices",
    ),
)
def _mppi_ik_solve_cuda_batch_jit(
    robot:                Robot,
    target_poses_batch:   jaxlie.SE3,
    rng_key:              Array,
    previous_cfgs:        Float[Array, "n_problems n_act"],
    num_seeds:            int,
    n_particles:          int,
    n_mppi_iters:         int,
    n_lbfgs_iters:        int,
    m_lbfgs:              int,
    pos_weight:           float,
    ori_weight:           float,
    sigma:                float,
    mppi_temperature:     float,
    eps_pos:              float,
    eps_ori:              float,
    continuity_weight:    float,
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
    from ..cuda_kernels._mppi_ik_cuda import mppi_ik_cuda

    n_act      = robot.joints.num_actuated_joints
    lower      = robot.joints.lower_limits
    upper      = robot.joints.upper_limits
    n_problems = previous_cfgs.shape[0]

    target_T_batch = target_poses_batch.wxyz_xyz.astype(jnp.float32)[:, None, :]

    n_warm   = max(1, num_seeds // 2)
    n_random = num_seeds - n_warm

    key_warm, key_random, key_seed = jax.random.split(rng_key, 3)

    warm_seeds = jnp.clip(
        previous_cfgs[:, None, :] + jax.random.normal(key_warm, (n_problems, n_warm, n_act)) * 0.05,
        lower, upper,
    )
    warm_seeds = jnp.where(fixed_joint_mask_int[None, None, :], previous_cfgs[:, None, :], warm_seeds)

    random_seeds = jax.random.uniform(
        key_random, (n_problems, n_random, n_act), minval=lower, maxval=upper
    )
    random_seeds = jnp.where(
        fixed_joint_mask_int[None, None, :], previous_cfgs[:, None, :], random_seeds
    )
    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=1)

    rng_seed_arr = jax.random.bits(key_seed, dtype=jnp.uint32).astype(jnp.int32)

    cfgs, errors = mppi_ik_cuda(
        seeds          = seeds,
        twists         = robot.joints.twists,
        parent_tf      = robot.joints.parent_transforms,
        parent_idx     = robot.joints.parent_indices,
        act_idx        = robot.joints.actuated_indices,
        mimic_mul      = robot.joints.mimic_multiplier,
        mimic_off      = robot.joints.mimic_offset,
        mimic_act_idx  = robot.joints.mimic_act_indices,
        topo_inv       = robot.joints._topo_sort_inv,
        target_jnts    = target_jnts,
        ancestor_masks = ancestor_masks,
        target_T       = target_T_batch,
        robot_spheres_local = robot_spheres_local,
        robot_sphere_joint_idx = robot_sphere_joint_idx,
        world_spheres = world_spheres,
        world_capsules = world_capsules,
        world_boxes = world_boxes,
        world_halfspaces = world_halfspaces,
        lower          = lower,
        upper          = upper,
        fixed_mask     = fixed_joint_mask_int,
        rng_seed       = rng_seed_arr,
        n_particles      = n_particles,
        n_mppi_iters     = n_mppi_iters,
        n_lbfgs_iters    = n_lbfgs_iters,
        m_lbfgs          = m_lbfgs,
        sigma            = sigma,
        mppi_temperature = mppi_temperature,
        pos_weight       = pos_weight,
        ori_weight       = ori_weight,
        eps_pos          = eps_pos,
        eps_ori          = eps_ori,
        enable_collision = enable_collision,
        collision_weight = collision_weight,
        collision_margin = collision_margin,
    )

    if len(constraint_fns) > 0:
        flat_cfgs = cfgs.reshape(n_problems * num_seeds, n_act)

        def constraint_penalty(cfg):
            c_vals = jnp.stack([
                constraint_fns[i](cfg, robot, constraint_args[i])
                for i in range(len(constraint_fns))
            ])
            return jnp.sum(constraint_weights * c_vals ** 2)

        flat_cpen    = jax.vmap(constraint_penalty)(flat_cfgs)
        cpen         = flat_cpen.reshape(n_problems, num_seeds)
        final_errors = (
            errors
            + cpen
            + continuity_weight * jnp.sum((cfgs - previous_cfgs[:, None, :]) ** 2, axis=-1)
        )
    else:
        final_errors = (
            errors
            + continuity_weight * jnp.sum((cfgs - previous_cfgs[:, None, :]) ** 2, axis=-1)
        )

    best_idxs = jnp.argmin(final_errors, axis=1)
    return cfgs[jnp.arange(n_problems), best_idxs]


def mppi_ik_solve_cuda_batch(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3,
    rng_key:             Array,
    previous_cfgs:       Float[Array, "n_problems n_act"],
    num_seeds:           int   = 32,
    n_particles:         int   = 16,
    n_mppi_iters:        int   = 5,
    n_lbfgs_iters:       int   = 25,
    m_lbfgs:             int   = 5,
    pos_weight:          float = 50.0,
    ori_weight:          float = 10.0,
    sigma:               float = 0.3,
    mppi_temperature:    float = 0.05,
    eps_pos:             float = 1e-8,
    eps_ori:             float = 1e-8,
    continuity_weight:   float = 0.0,
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
) -> Float[Array, "n_problems n_act"]:
    """Batched CUDA MPPI+L-BFGS IK: solve n_problems targets in one kernel launch.

    Args:
        robot:               The robot model.
        target_link_indices: Index (or tuple of indices) of target link(s).
        target_poses:        Batch of SE(3) targets, shape ``(n_problems,)``.
        rng_key:             JAX PRNG key.
        previous_cfgs:       Previous configurations, shape ``(n_problems, n_act)``.
        num_seeds:           Parallel seeds per problem.
        n_particles:         MPPI particles per step.
        n_mppi_iters:        MPPI stage iterations.
        n_lbfgs_iters:       L-BFGS stage iterations.
        m_lbfgs:             L-BFGS history size.
        pos_weight:          Weight on position residual.
        ori_weight:          Weight on orientation residual.
        sigma:               MPPI noise std dev [rad/m].
        mppi_temperature:    MPPI softmax temperature.
        eps_pos:             Position convergence threshold [m].
        eps_ori:             Orientation convergence threshold [rad].
        continuity_weight:   Weight on ‖q − prev‖² in winner selection.
        fixed_joint_mask:    Boolean mask; True = joint must not move.
        constraints:         List of constraint callables.
        constraint_weights:  Scalar weight per constraint.

    Returns:
        Best joint configurations, shape ``(n_problems, n_act)``.
    """
    if isinstance(target_link_indices, int):
        target_link_indices = (target_link_indices,)

    n_act = robot.joints.num_actuated_joints

    if fixed_joint_mask is None:
        fixed_joint_mask_int = jnp.zeros(n_act, dtype=jnp.int32)
    else:
        fixed_joint_mask_int = fixed_joint_mask.astype(jnp.int32)

    (
        cuda_constraint_fns,
        cuda_constraint_args,
        cuda_constraint_weights,
        _post_constraint_fns,
        _post_constraint_args,
        _post_constraint_weights,
    ) = split_cuda_and_post_constraints(
        constraints=constraints,
        constraint_args=constraint_args,
        constraint_weights=constraint_weights,
        collision_constraint_indices=collision_constraint_indices,
        collision_free=collision_free,
    )

    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    parent_idx_np           = np.array(robot.joints.parent_indices)
    n_joints                = robot.joints.num_joints
    target_joint_idx        = int(parent_joint_indices_np[target_link_indices[0]])
    ancestor_mask_np        = np.zeros(n_joints, dtype=np.int32)
    j = target_joint_idx
    while j >= 0:
        ancestor_mask_np[j] = 1
        j = int(parent_idx_np[j])
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

    return _mppi_ik_solve_cuda_batch_jit(
        robot=robot,
        target_poses_batch=target_poses,
        rng_key=rng_key,
        previous_cfgs=previous_cfgs,
        num_seeds=num_seeds,
        n_particles=n_particles,
        n_mppi_iters=n_mppi_iters,
        n_lbfgs_iters=n_lbfgs_iters,
        m_lbfgs=m_lbfgs,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        sigma=sigma,
        mppi_temperature=mppi_temperature,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        continuity_weight=continuity_weight,
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
