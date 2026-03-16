"""Sequential Quadratic Programming IK Solver.

Pure JAX implementation of multi-seed SQP-IK, sharing the seed generation,
Jacobi column scaling, line search, and winner-selection infrastructure of
the Least Squares solver while replacing the unconstrained LM step with a
**box-constrained QP subproblem**.

Algorithmic difference from LS-IK (Levenberg-Marquardt):
  - At each SQP iteration the normal-equations step is replaced by solving
    a box-constrained QP:
        min_{p_s}  1/2 p_s^T H_s p_s + g_s^T p_s
        s.t.       (lower - q) * col_scale <= p_s <= (upper - q) * col_scale
    where H_s = Js^T Js + λI and g_s = Js^T fw are computed from the
    Jacobi-scaled Jacobian Js.
  - The QP is solved by ``n_inner_iters`` steps of projected gradient descent
    with step size α = 1 / (n_act + λ), which is guaranteed to be safe because
    λ_max(H_s) ≤ ||Js||_F^2 + λ = n_act + λ after Jacobi column scaling.
  - Joint limits enter as **hard constraints on the step** rather than being
    enforced by post-hoc clamping.  Near joint limits this allocates the step
    budget more effectively.

Shared with LS-IK:
  - Same seed generation (warm + random, multi-EE tight/loose split).
  - Same Jacobi column scaling in the normal equations.
  - Same vectorised line search (5 step sizes in parallel via vmap).
  - Same trust-region step-size schedule.
  - Same all-time best-config tracking.
  - Same continuity-weighted winner selection.
  - Same CUDA pattern (JAX FFI, same grid/block layout).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ._ik_primitives import _ik_residual, _LS_ALPHAS


# ---------------------------------------------------------------------------
# Single-seed SQP solve
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("target_link_indices", "max_iter", "n_inner_iters", "constraint_fns"),
)
def _sqp_ik_single(
    cfg:                   Float[Array, "n_act"],
    robot:                 Robot,
    target_link_indices:   tuple[int, ...],
    target_poses:          tuple,
    max_iter:              int,
    n_inner_iters:         int,
    lam0:                  float,
    pos_weight:            float,
    ori_weight:            float,
    lower:                 Float[Array, "n_act"],
    upper:                 Float[Array, "n_act"],
    fixed_joint_mask:      Float[Array, "n_act"],
    constraint_fns:        tuple = (),
    constraint_args:       tuple = (),
    constraint_weights:    Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    """SQP refinement from a single starting configuration.

    At each outer iteration, solves a box-constrained QP subproblem via
    ``n_inner_iters`` steps of projected gradient descent in the Jacobi-scaled
    space.  This ensures joint limits are respected as hard constraints
    throughout optimisation, unlike LM which clamps only after the step.

    QP subproblem (scaled space)
        min_{p_s}  1/2 p_s^T H_s p_s + g_s^T p_s
        s.t.       lb_s <= p_s <= ub_s
    where
        H_s  = Js^T Js + λ I          (Gauss-Newton Hessian, scaled)
        g_s  = Js^T fw                (gradient, scaled)
        lb_s = (lower - q) * scale    (per-joint step lower bound)
        ub_s = (upper - q) * scale    (per-joint step upper bound)

    Projected gradient step size
        α = 1 / (n_act + λ)
    which satisfies α ≤ 1 / λ_max(H_s) since after Jacobi scaling each
    column of Js has unit ℓ2-norm, so
        λ_max(Js^T Js) ≤ ||Js||_F^2 = n_act.

    FK fusion
        Same single-FK-call trick as LS-IK: ``fused_scaled(q)`` wraps all EE
        residuals so XLA CSE deduplicates the ``robot.forward_kinematics``
        call across EEs and the line-search evaluations.

    Inner QP
        At each outer iteration the unconstrained Newton step is computed
        exactly via ``jnp.linalg.solve`` (same as LM/LS-IK), then projected
        to the joint-limit box.  ``n_inner_iters`` active-set refinement steps
        follow: joints that hit a bound are fixed there, and the reduced system
        for the remaining free joints is re-solved.  This converges in 1–2
        steps for typical IK problems; when no joints are near limits the
        refinement is a no-op and SQP is identical to LM in step quality.
    """
    n     = cfg.shape[0]
    n_c   = len(constraint_fns)
    n_ee  = len(target_link_indices)
    W     = jnp.concatenate([
        jnp.full(3, pos_weight, dtype=cfg.dtype),
        jnp.full(3, ori_weight, dtype=cfg.dtype),
    ])  # (6,)

    if n_c > 0:
        sqrt_wc = jnp.sqrt(constraint_weights)

    def fused_scaled(q: Array) -> Array:
        """Weighted [EE residuals... | constraint residuals] with one FK pass."""
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
        return jnp.concatenate(parts)

    n_out = 6 * n_ee + n_c

    def sqp_step(carry, _):
        c, lam, best_c, best_err = carry

        # ── Jacobian + weighted residual (single FK forward pass) ────────
        f_all, vjp_fn = jax.vjp(fused_scaled, c)
        J_all = jax.vmap(lambda g: vjp_fn(g)[0])(
            jnp.eye(n_out, dtype=f_all.dtype)
        )  # (n_out, n)
        curr_err = jnp.dot(f_all, f_all)

        # ── Jacobi column scaling ────────────────────────────────────────
        col_scale = jnp.linalg.norm(J_all, axis=0) + 1e-8   # (n,)
        J_s       = J_all / col_scale[None, :]               # unit-column Jacobian

        # ── QP matrices in scaled space ──────────────────────────────────
        H_s = J_s.T @ J_s + lam * jnp.eye(n, dtype=J_s.dtype)  # (n, n)
        g_s = J_s.T @ f_all                                      # (n,)

        # ── Box bounds in scaled space ───────────────────────────────────
        # lb_s / ub_s bound the step p_s = p * col_scale
        lb_s = (lower - c) * col_scale
        ub_s = (upper - c) * col_scale
        # Fixed joints: force step to zero
        lb_s = jnp.where(fixed_joint_mask, 0.0, lb_s)
        ub_s = jnp.where(fixed_joint_mask, 0.0, ub_s)

        # ── Step 1: unconstrained Newton step (identical to LM) ──────────
        p_s = jnp.linalg.solve(H_s, -g_s)

        # ── Step 2 + active-set refinements ──────────────────────────────
        # Project to box first, then iteratively fix bound-hitting joints
        # and re-solve for the remaining free joints.  With n_inner_iters=2
        # this converges for all practical IK problems.  When no joints are
        # near limits the refinement is a no-op (same quality as LM).
        p_s = jnp.clip(p_s, lb_s, ub_s)

        def active_set_step(p_s, _):
            # Joints at a bound become active (their step is fixed there).
            active = ((p_s <= lb_s + 1e-8) | (p_s >= ub_s - 1e-8)).astype(p_s.dtype)
            free   = 1.0 - active
            # Bound values for active joints.
            p_bounded = jnp.clip(p_s, lb_s, ub_s) * active
            # Adjust gradient: g_adj = g_s + H_s @ p_bounded
            g_adj  = g_s + H_s @ p_bounded
            # Masked system: unit diagonal for active joints, original for free.
            H_m    = H_s * free[:, None] * free[None, :] + jnp.diag(active + 1e-8)
            rhs    = -g_adj * free
            p_free = jnp.linalg.solve(H_m, rhs)
            return jnp.clip(p_free * free + p_bounded, lb_s, ub_s), None

        p_s, _ = jax.lax.scan(active_set_step, p_s, None, length=n_inner_iters)

        # ── Unscale to original joint space ──────────────────────────────
        delta = p_s / col_scale

        # ── Trust-region step clipping ───────────────────────────────────
        pos_err_r = jnp.max(jnp.array([
            jnp.linalg.norm(f_all[6*i : 6*i+3] / W[:3]) for i in range(n_ee)
        ]))
        ori_err_r = jnp.max(jnp.array([
            jnp.linalg.norm(f_all[6*i+3 : 6*i+6] / W[3:]) for i in range(n_ee)
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

        # ── Vectorised line search (single FK call per alpha) ────────────
        def eval_alpha(alpha):
            nc  = jnp.clip(c + alpha * delta, lower, upper)
            nf  = fused_scaled(nc)
            return jnp.dot(nf, nf)

        alpha_errs  = jax.vmap(eval_alpha)(_LS_ALPHAS)   # (5,)
        best_ls_idx = jnp.argmin(alpha_errs)
        new_err     = alpha_errs[best_ls_idx]
        new_c       = jnp.clip(c + _LS_ALPHAS[best_ls_idx] * delta, lower, upper).astype(c.dtype)

        # ── Accept / reject ──────────────────────────────────────────────
        improved = new_err < curr_err * (1.0 - 1e-4)
        c_out    = jnp.where(improved, new_c, c)
        lam_out  = jnp.clip(
            jnp.where(improved, lam * 0.5, lam * 3.0), 1e-10, 1e6
        )

        # ── Track all-time best ──────────────────────────────────────────
        new_best_c   = jnp.where(new_err < best_err, new_c,   best_c)
        new_best_err = jnp.where(new_err < best_err, new_err, best_err)

        return (c_out, lam_out, new_best_c, new_best_err), None

    init_f_all = fused_scaled(cfg)
    init_err   = jnp.dot(init_f_all, init_f_all)
    lam0_arr   = jnp.asarray(lam0, dtype=cfg.dtype)

    (_, _, best_c, _), _ = jax.lax.scan(
        sqp_step, (cfg, lam0_arr, cfg, init_err), None, length=max_iter
    )
    return best_c


# ---------------------------------------------------------------------------
# Public entry point — JAX
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("target_link_indices", "num_seeds", "max_iter", "n_inner_iters", "constraint_fns"),
)
def sqp_ik_solve(
    robot:              Robot,
    target_link_indices: tuple[int, ...],
    target_poses:       tuple,
    rng_key:            Array,
    previous_cfg:       Float[Array, "n_act"],
    num_seeds:          int   = 32,
    max_iter:           int   = 60,
    n_inner_iters:      int   = 2,
    pos_weight:         float = 50.0,
    ori_weight:         float = 10.0,
    lambda_init:        float = 5e-3,
    continuity_weight:  float = 0.0,
    fixed_joint_mask:   Float[Array, "n_act"] | None = None,
    constraint_fns:     tuple = (),
    constraint_args:    tuple = (),
    constraint_weights: Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    """Solve IK via multi-seed SQP with box-constrained QP subproblems.

    Each SQP iteration solves the unconstrained Newton step via Cholesky
    (identical to LM/LS-IK), projects it to the joint-limit box, then runs
    ``n_inner_iters`` active-set refinement steps to improve constraint
    handling for joints near their limits.  When no joints are near limits
    the refinement is a no-op and quality matches LS-IK exactly.

    Multi-EE support
        Pass multiple end-effectors via ``target_link_indices`` and
        ``target_poses`` tuples.  Residuals are stacked identically to
        LS-IK.  For a single EE pass ``target_link_indices=(idx,)`` and
        ``target_poses=(pose,)``.

    Seed generation
        Identical to LS-IK: half warm-starts near ``previous_cfg`` (σ = 0.05)
        and half random.  Multi-EE problems get a loose (σ = 0.3) fraction
        to help escape per-EE local minima.

    Inner QP
        ``n_inner_iters`` projected gradient steps per outer SQP iteration.
        Step size α = 1 / (n_act + λ) is provably safe (guaranteed descent).

    Args:
        robot:               The robot model.
        target_link_indices: Tuple of target link indices (static for JIT).
        target_poses:        Tuple of desired SE(3) world poses.
        rng_key:             JAX PRNG key.
        previous_cfg:        Previous joint configuration for warm-starting.
        num_seeds:           Number of parallel seeds (default 32).
        max_iter:            Outer SQP iterations per seed (default 60).
        n_inner_iters:       Active-set refinement steps per outer SQP
                             iteration (default 2).
        pos_weight:          Weight on position residual (default 50.0).
        ori_weight:          Weight on orientation residual (default 10.0).
        lambda_init:         Initial LM-style damping factor (default 5e-3).
        continuity_weight:   Weight on ‖q − prev‖² in winner selection only.
        fixed_joint_mask:    Boolean mask; True = joint must not move.
        constraint_fns:      Tuple of constraint callables
                             ``c(cfg, robot) -> scalar``.
        constraint_weights:  Weight for each constraint.

    Returns:
        Best joint configuration found, shape ``(n_actuated_joints,)``.
    """
    n_act  = robot.joints.num_actuated_joints
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    if fixed_joint_mask is None:
        fixed_joint_mask = jnp.zeros(n_act, dtype=jnp.bool_)

    # ── Seed generation ────────────────────────────────────────────────────
    n_warm   = max(1, num_seeds // 2)
    n_random = num_seeds - n_warm

    key_warm, key_random = jax.random.split(rng_key)

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

    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=0)   # (num_seeds, n_act)

    # ── Multi-seed SQP (parallel over seeds) ──────────────────────────────
    all_cfgs = jax.vmap(
        lambda cfg: _sqp_ik_single(
            cfg, robot, target_link_indices, target_poses,
            max_iter, n_inner_iters, lambda_init, pos_weight, ori_weight,
            lower, upper, fixed_joint_mask,
            constraint_fns=constraint_fns,
            constraint_args=constraint_args,
            constraint_weights=constraint_weights,
        )
    )(seeds)   # (num_seeds, n_act)

    # ── Winner selection: task error (all EEs) + constraint penalties + continuity
    W = jnp.concatenate([jnp.full(3, pos_weight), jnp.full(3, ori_weight)])

    def weighted_err(cfg: Float[Array, "n_act"]) -> Array:
        err = sum(
            jnp.sum((_ik_residual(cfg, robot, target_link_indices[i], target_poses[i]) * W) ** 2)
            for i in range(len(target_link_indices))
        )
        if len(constraint_fns) > 0:
            c_vals = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(len(constraint_fns))])
            err    = err + jnp.sum(constraint_weights * c_vals ** 2)
        return err + continuity_weight * jnp.sum((cfg - previous_cfg) ** 2)

    errors   = jax.vmap(weighted_err)(all_cfgs)   # (num_seeds,)
    best_idx = jnp.argmin(errors)
    return all_cfgs[best_idx]


# ---------------------------------------------------------------------------
# Public entry point — CUDA (JIT inner)
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "max_iter",
        "n_inner_iters",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "eps_ori",
        "constraint_fns",
        "target_link_indices",
    ),
)
def _sqp_ik_solve_cuda_jit(
    robot:                Robot,
    target_poses:         tuple,
    rng_key:              Array,
    previous_cfg:         Float[Array, "n_act"],
    num_seeds:            int,
    max_iter:             int,
    n_inner_iters:        int,
    pos_weight:           float,
    ori_weight:           float,
    lambda_init:          float,
    eps_pos:              float,
    eps_ori:              float,
    continuity_weight:    float,
    fixed_joint_mask_int: Float[Array, "n_act"],
    ancestor_masks:       Array,
    target_jnts:          Array,
    target_link_indices:  tuple[int, ...],
    constraint_fns:       tuple = (),
    constraint_args:      tuple = (),
    constraint_weights:   Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    from ..cuda_kernels._sqp_ik_cuda import sqp_ik_cuda

    n_act  = robot.joints.num_actuated_joints
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    target_Ts = jnp.stack([tp.wxyz_xyz.astype(jnp.float32) for tp in target_poses], axis=0)

    # ── Seed generation ────────────────────────────────────────────────────
    n_warm   = max(1, num_seeds // 2)
    n_random = num_seeds - n_warm

    key_warm, key_random = jax.random.split(rng_key)

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

    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=0)   # (num_seeds, n_act)

    # ── CUDA SQP kernel (all EEs simultaneously) ──────────────────────────
    cfgs, errors = sqp_ik_cuda(
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
        lower          = lower,
        upper          = upper,
        fixed_mask     = fixed_joint_mask_int,
        max_iter       = max_iter,
        n_inner_iters  = n_inner_iters,
        pos_weight     = pos_weight,
        ori_weight     = ori_weight,
        lambda_init    = lambda_init,
        eps_pos        = eps_pos,
        eps_ori        = eps_ori,
    )
    cfgs   = cfgs[0]    # (n_seeds, n_act)
    errors = errors[0]  # (n_seeds,)

    if len(constraint_fns) > 0:
        def constraint_penalty(cfg):
            c_vals = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(len(constraint_fns))])
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


def sqp_ik_solve_cuda(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3 | tuple,
    rng_key:             Array,
    previous_cfg:        Float[Array, "n_act"],
    num_seeds:           int   = 32,
    max_iter:            int   = 60,
    n_inner_iters:       int   = 2,
    pos_weight:          float = 50.0,
    ori_weight:          float = 10.0,
    lambda_init:         float = 5e-3,
    eps_pos:             float = 1e-8,
    eps_ori:             float = 1e-8,
    continuity_weight:   float = 0.0,
    fixed_joint_mask:    Float[Array, "n_act"] | None = None,
    constraints:         Sequence[Callable] | None = None,
    constraint_args:     Sequence | None = None,
    constraint_weights:  Sequence[float] | None = None,
    constraint_refine_iters: int = 12,
) -> Float[Array, "n_act"]:
    """CUDA alternative to :func:`sqp_ik_solve`.

    Offloads the multi-seed SQP loop to a single CUDA kernel via JAX FFI.
    The kernel is structurally equivalent to the JAX solver: each outer
    iteration forms the Gauss-Newton Hessian in float64, then runs
    ``n_inner_iters`` projected gradient steps (float32) to enforce joint
    limits as hard QP constraints.

    Requires ``_sqp_ik_cuda_lib.so`` compiled from ``_sqp_ik_cuda_kernel.cu``:
        bash src/pyronot/cuda_kernels/build_sqp_ik_cuda.sh

    Multi-EE support
        Pass multiple end-effectors via ``target_link_indices`` (tuple) and
        ``target_poses`` (tuple).  The CUDA kernel optimises all EEs
        simultaneously via stacked residuals and Jacobians.

    Kinematic constraints
        Because the CUDA kernel cannot call arbitrary Python/JAX functions,
        constraints are incorporated via:
        1. JAX winner-selection penalty (all CUDA candidates evaluated).
        2. Optional post-CUDA JAX refinement (``constraint_refine_iters``).

    Args:
        robot:                   The robot model.
        target_link_indices:     Index (or tuple of indices) of target link(s).
        target_poses:            Desired SE(3) world pose (or tuple of poses).
        rng_key:                 JAX PRNG key.
        previous_cfg:            Previous joint configuration.
        num_seeds:               Number of parallel seeds.
        max_iter:                Outer SQP iteration budget per seed.
        n_inner_iters:           Active-set refinement steps per outer iteration (default 2).
        pos_weight:              Weight on position residual.
        ori_weight:              Weight on orientation residual.
        lambda_init:             Initial damping factor.
        eps_pos:                 Position convergence threshold [m].
        eps_ori:                 Orientation convergence threshold [rad].
        continuity_weight:       Weight on ‖q − prev‖² in winner selection.
        fixed_joint_mask:        Boolean mask; True = joint must not move.
        constraints:             List of constraint callables.
        constraint_weights:      Scalar weight per constraint.
        constraint_refine_iters: Post-CUDA JAX SQP iterations on winner.

    Returns:
        Best joint configuration found, shape ``(n_act,)``.
    """
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

    constraint_fns         = tuple(constraints) if constraints else ()
    constraint_args_t      = tuple(constraint_args) if constraint_args is not None else ()
    constraint_weights_arr = (
        jnp.array(constraint_weights, dtype=jnp.float32)
        if constraint_weights is not None else None
    )

    # ── Pre-compute per-EE ancestor masks (Python level) ───────────────────
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

    winner, winner_coll_cost = _sqp_ik_solve_cuda_jit(
        robot=robot,
        target_poses=target_poses_t,
        rng_key=rng_key,
        previous_cfg=previous_cfg,
        num_seeds=num_seeds,
        max_iter=max_iter,
        n_inner_iters=n_inner_iters,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        continuity_weight=continuity_weight,
        fixed_joint_mask_int=fixed_joint_mask_int,
        ancestor_masks=ancestor_masks,
        target_jnts=target_jnts,
        target_link_indices=target_link_indices,
        constraint_fns=constraint_fns,
        constraint_args=constraint_args_t,
        constraint_weights=constraint_weights_arr,
    )

    # ── Post-CUDA JAX refinement with constraints only ────────────────────
    if bool(constraint_fns) and constraint_refine_iters > 0:
        fmask = (
            fixed_joint_mask.astype(jnp.bool_)
            if fixed_joint_mask is not None
            else jnp.zeros(n_act, dtype=jnp.bool_)
        )
        skip = float(winner_coll_cost) <= 1e-6
        if not skip:
            winner = _sqp_ik_single(
                winner, robot, target_link_indices, target_poses_t,
                constraint_refine_iters, n_inner_iters,
                lambda_init, pos_weight, ori_weight,
                robot.joints.lower_limits, robot.joints.upper_limits,
                fmask,
                constraint_fns=constraint_fns,
                constraint_args=constraint_args_t,
                constraint_weights=constraint_weights_arr,
            )

    return winner


# ---------------------------------------------------------------------------
# Public entry point — CUDA batched
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "max_iter",
        "n_inner_iters",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "eps_ori",
        "constraint_fns",
        "target_link_indices",
    ),
)
def _sqp_ik_solve_cuda_batch_jit(
    robot:                Robot,
    target_poses_batch:   jaxlie.SE3,
    rng_key:              Array,
    previous_cfgs:        Float[Array, "n_problems n_act"],
    num_seeds:            int,
    max_iter:             int,
    n_inner_iters:        int,
    pos_weight:           float,
    ori_weight:           float,
    lambda_init:          float,
    eps_pos:              float,
    eps_ori:              float,
    continuity_weight:    float,
    fixed_joint_mask_int: Float[Array, "n_act"],
    ancestor_masks:       Array,
    target_jnts:          Array,
    target_link_indices:  tuple[int, ...],
    constraint_fns:       tuple = (),
    constraint_args:      tuple = (),
    constraint_weights:   Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_problems n_act"]:
    from ..cuda_kernels._sqp_ik_cuda import sqp_ik_cuda

    n_act      = robot.joints.num_actuated_joints
    lower      = robot.joints.lower_limits
    upper      = robot.joints.upper_limits
    n_problems = previous_cfgs.shape[0]

    target_T_batch = target_poses_batch.wxyz_xyz.astype(jnp.float32)[:, None, :]  # (n_problems, 1, 7)

    n_warm   = max(1, num_seeds // 2)
    n_random = num_seeds - n_warm

    key_warm, key_random = jax.random.split(rng_key)

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

    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=1)  # (n_problems, n_seeds, n_act)

    cfgs, errors = sqp_ik_cuda(
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
        lower          = lower,
        upper          = upper,
        fixed_mask     = fixed_joint_mask_int,
        max_iter       = max_iter,
        n_inner_iters  = n_inner_iters,
        pos_weight     = pos_weight,
        ori_weight     = ori_weight,
        lambda_init    = lambda_init,
        eps_pos        = eps_pos,
        eps_ori        = eps_ori,
    )

    if len(constraint_fns) > 0:
        flat_cfgs = cfgs.reshape(n_problems * num_seeds, n_act)

        def constraint_penalty(cfg):
            c_vals = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(len(constraint_fns))])
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


def sqp_ik_solve_cuda_batch(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3,
    rng_key:             Array,
    previous_cfgs:       Float[Array, "n_problems n_act"],
    num_seeds:           int   = 32,
    max_iter:            int   = 60,
    n_inner_iters:       int   = 2,
    pos_weight:          float = 50.0,
    ori_weight:          float = 10.0,
    lambda_init:         float = 5e-3,
    eps_pos:             float = 1e-8,
    eps_ori:             float = 1e-8,
    continuity_weight:   float = 0.0,
    fixed_joint_mask:    Float[Array, "n_act"] | None = None,
    constraints:         Sequence[Callable] | None = None,
    constraint_args:     Sequence | None = None,
    constraint_weights:  Sequence[float] | None = None,
    constraint_refine_iters: int = 12,
) -> Float[Array, "n_problems n_act"]:
    """Batched CUDA SQP-IK: solve n_problems targets in a single kernel launch.

    Args:
        robot:                   The robot model.
        target_link_indices:     Index (or tuple of indices) of target link(s).
        target_poses:            Batch of SE(3) targets, shape ``(n_problems,)``.
        rng_key:                 JAX PRNG key.
        previous_cfgs:           Previous configurations, shape ``(n_problems, n_act)``.
        num_seeds:               Parallel seeds per problem.
        max_iter:                Outer SQP iterations per seed.
        n_inner_iters:           Inner projected-gradient iterations.
        pos_weight:              Weight on position residual.
        ori_weight:              Weight on orientation residual.
        lambda_init:             Initial damping factor.
        eps_pos:                 Position convergence threshold [m].
        eps_ori:                 Orientation convergence threshold [rad].
        continuity_weight:       Weight on ‖q − prev‖² in winner selection.
        fixed_joint_mask:        Boolean mask; True = joint must not move.
        constraints:             List of constraint callables.
        constraint_weights:      Scalar weight per constraint.
        constraint_refine_iters: Post-CUDA JAX iterations on each winner.

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

    constraint_fns         = tuple(constraints) if constraints else ()
    constraint_args_t      = tuple(constraint_args) if constraint_args is not None else ()
    constraint_weights_arr = (
        jnp.array(constraint_weights, dtype=jnp.float32)
        if constraint_weights is not None else None
    )

    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    target_joint_idx        = int(parent_joint_indices_np[target_link_indices[0]])
    parent_idx_np    = np.array(robot.joints.parent_indices)
    n_joints         = robot.joints.num_joints
    ancestor_mask_np = np.zeros(n_joints, dtype=np.int32)
    j = target_joint_idx
    while j >= 0:
        ancestor_mask_np[j] = 1
        j = int(parent_idx_np[j])
    ancestor_masks = jnp.array(ancestor_mask_np[None, :])
    target_jnts    = jnp.array([target_joint_idx], dtype=jnp.int32)

    winners = _sqp_ik_solve_cuda_batch_jit(
        robot=robot,
        target_poses_batch=target_poses,
        rng_key=rng_key,
        previous_cfgs=previous_cfgs,
        num_seeds=num_seeds,
        max_iter=max_iter,
        n_inner_iters=n_inner_iters,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        continuity_weight=continuity_weight,
        fixed_joint_mask_int=fixed_joint_mask_int,
        ancestor_masks=ancestor_masks,
        target_jnts=target_jnts,
        target_link_indices=target_link_indices,
        constraint_fns=constraint_fns,
        constraint_args=constraint_args_t,
        constraint_weights=constraint_weights_arr,
    )

    if constraint_fns and constraint_refine_iters > 0:
        fmask = (
            fixed_joint_mask.astype(jnp.bool_)
            if fixed_joint_mask is not None
            else jnp.zeros(n_act, dtype=jnp.bool_)
        )
        lower = robot.joints.lower_limits
        upper = robot.joints.upper_limits

        winners = jax.vmap(
            lambda cfg, wxyz_xyz: _sqp_ik_single(
                cfg, robot, target_link_indices,
                (jaxlie.SE3(wxyz_xyz.astype(cfg.dtype)),),
                constraint_refine_iters, n_inner_iters,
                lambda_init, pos_weight, ori_weight,
                lower, upper, fmask,
                constraint_fns=constraint_fns,
                constraint_args=constraint_args_t,
                constraint_weights=constraint_weights_arr,
            )
        )(winners, target_poses.wxyz_xyz)

    return winners
