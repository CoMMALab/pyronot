"""Gauss-Newton Least Squares IK Solver.

Pure JAX implementation of multi-seed Levenberg-Marquardt IK, structured
in the same spirit as jaxls (clean Gauss-Newton normal equations, trust-region
lambda schedule) but without any jaxls dependency.

Differences from the HJCD-IK solver (_hjcd_ik.py):
  - No coarse Hamiltonian coordinate-descent phase.  All seeds go directly
    into Levenberg-Marquardt.
  - Fixed pos_weight / ori_weight instead of adaptive row-equilibration.
  - No stall detection or random kicks.
  - No joint-limit prior in the normal equations (hard clamping only).
  - Full-batch Gauss-Newton update per step (vs. single-joint CD step).

Shared with HJCD-IK:
  - Same seed generation (warm + random).
  - Same Jacobi column scaling in the normal equations.
  - Same vectorised line search (5 step sizes evaluated in parallel).
  - Same trust-region step-size schedule.
  - Same all-time best-config tracking.
  - Same continuity-weighted winner selection.
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
from ._ik_primitives import _ik_residual, _LS_ALPHAS

# ---------------------------------------------------------------------------
# Single-seed LM solve
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("target_link_index", "max_iter"),
)
def _ls_ik_single(
    cfg:              Float[Array, "n_act"],
    robot:            Robot,
    target_link_index: int,
    target_pose:      jaxlie.SE3,
    max_iter:         int,
    lam0:             float,
    pos_weight:       float,
    ori_weight:       float,
    lower:            Float[Array, "n_act"],
    upper:            Float[Array, "n_act"],
    fixed_joint_mask: Float[Array, "n_act"],
) -> Float[Array, "n_act"]:
    """Levenberg-Marquardt refinement from a single starting configuration.

    Normal equations
        (Js^T Js + lambda * I) p = -Js^T fw
    where Js is the column-scaled weighted Jacobian and fw = W * f.

    Float64 solve
        The normal-equation matrix is formed and solved in float64, matching
        the HJCD-IK LM phase for a fair comparison.

    Line search
        Five step-size candidates {1, 0.5, 0.25, 0.1, 0.025} are evaluated
        in parallel via vmap; the one producing the lowest weighted residual
        is accepted.
    """
    n   = cfg.shape[0]
    W   = jnp.concatenate([
        jnp.full(3, pos_weight, dtype=cfg.dtype),
        jnp.full(3, ori_weight, dtype=cfg.dtype),
    ])  # (6,)

    def lm_step(carry, _):
        c, lam, best_c, best_err = carry

        # ── Jacobian + residual ──────────────────────────────────────────
        residual_fn = lambda q: _ik_residual(q, robot, target_link_index, target_pose)
        f, vjp_fn = jax.vjp(residual_fn, c)
        J = jax.vmap(lambda g: vjp_fn(g)[0])(jnp.eye(6, dtype=c.dtype))  # (6, n)

        fw       = f * W                          # weighted residual
        Jw       = J * W[:, None]                 # weighted Jacobian  (6, n)
        curr_err = jnp.dot(fw, fw)

        # ── Jacobi column scaling ────────────────────────────────────────
        col_scale = jnp.linalg.norm(Jw, axis=0) + 1e-8   # (n,)
        Js        = Jw / col_scale[None, :]               # unit-column Jacobian

        # ── Normal equations with LM damping (float64) ──────────────────
        Js_d  = Js.astype(jnp.float64)
        fw_d  = fw.astype(jnp.float64)
        lam_d = lam.astype(jnp.float64)

        A   = Js_d.T @ Js_d + lam_d * jnp.eye(n, dtype=jnp.float64)
        rhs = -(Js_d.T @ fw_d)
        p   = jnp.linalg.solve(A, rhs).astype(c.dtype)   # (n,) in scaled space

        # Unscale and freeze fixed joints.
        delta = p / col_scale
        delta = jnp.where(fixed_joint_mask, 0.0, delta)

        # ── Trust-region step clipping ───────────────────────────────────
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

        # ── Vectorised line search ───────────────────────────────────────
        def eval_alpha(alpha):
            nc = jnp.clip(c + alpha * delta, lower, upper)
            nf = _ik_residual(nc, robot, target_link_index, target_pose)
            nfw = nf * W
            return jnp.dot(nfw, nfw)

        alpha_errs  = jax.vmap(eval_alpha)(_LS_ALPHAS)   # (5,)
        best_ls_idx = jnp.argmin(alpha_errs)
        new_err     = alpha_errs[best_ls_idx]
        new_c       = jnp.clip(c + _LS_ALPHAS[best_ls_idx] * delta, lower, upper)

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

    init_f    = _ik_residual(cfg, robot, target_link_index, target_pose)
    init_err  = jnp.dot(init_f * W, init_f * W)
    lam0_arr  = jnp.asarray(lam0, dtype=cfg.dtype)

    (_, _, best_c, _), _ = jax.lax.scan(
        lm_step, (cfg, lam0_arr, cfg, init_err), None, length=max_iter
    )
    return best_c


# ---------------------------------------------------------------------------
# Public entry point — JAX
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("target_link_index", "num_seeds", "max_iter"),
)
def ls_ik_solve(
    robot:             Robot,
    target_link_index: int,
    target_pose:       jaxlie.SE3,
    rng_key:           Array,
    previous_cfg:      Float[Array, "n_act"],
    num_seeds:         int   = 32,
    max_iter:          int   = 60,
    pos_weight:        float = 50.0,
    ori_weight:        float = 10.0,
    lambda_init:       float = 5e-3,
    continuity_weight: float = 0.0,
    fixed_joint_mask:  Float[Array, "n_act"] | None = None,
) -> Float[Array, "n_act"]:
    """Solve IK via multi-seed Levenberg-Marquardt (no coarse phase).

    All ``num_seeds`` seeds are refined directly with LM — there is no
    Hamiltonian coordinate-descent coarse phase.  This makes the solver
    a clean Gauss-Newton least-squares method suitable for comparison with
    HJCD-IK.

    Seed generation
        Half the seeds are warm-starts near ``previous_cfg`` (σ = 0.05).
        The remaining half are drawn uniformly across joint limits.

    Per-seed solve
        Each seed gets ``max_iter`` LM iterations with Jacobi column
        scaling, a 5-point vectorised line search, and best-config tracking.
        All seeds run in parallel via ``jax.vmap``.

    Winner selection
        Same as ``hjcd_solve``: lowest weighted task-space residual plus
        optional ``continuity_weight · ‖q − prev‖²`` tie-breaker.

    Args:
        robot:              The robot model.
        target_link_index:  Index of the target link in ``robot.links.names``.
        target_pose:        Desired SE(3) world pose for the target link.
        rng_key:            JAX PRNG key.
        previous_cfg:       Previous joint configuration for warm-starting
                            and continuity-aware winner selection.
        num_seeds:          Number of parallel seeds (default 32).
        max_iter:           LM iterations per seed (default 60).
        pos_weight:         Weight on position residual (default 50.0).
        ori_weight:         Weight on orientation residual (default 10.0).
        lambda_init:        Initial LM damping factor (default 5e-3).
        continuity_weight:  Weight on ‖q − prev‖² in winner selection only.
        fixed_joint_mask:   Boolean mask; True = joint must not move.

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

    # ── Multi-seed LM (parallel over seeds) ───────────────────────────────
    all_cfgs = jax.vmap(
        lambda cfg: _ls_ik_single(
            cfg, robot, target_link_index, target_pose,
            max_iter, lambda_init, pos_weight, ori_weight,
            lower, upper, fixed_joint_mask,
        )
    )(seeds)   # (num_seeds, n_act)

    # ── Winner selection: task error + continuity tie-breaker ──────────────
    W = jnp.concatenate([jnp.full(3, pos_weight), jnp.full(3, ori_weight)])

    def weighted_err(cfg: Float[Array, "n_act"]) -> Array:
        f = _ik_residual(cfg, robot, target_link_index, target_pose)
        fw = f * W
        return jnp.dot(fw, fw) + continuity_weight * jnp.sum((cfg - previous_cfg) ** 2)

    errors   = jax.vmap(weighted_err)(all_cfgs)   # (num_seeds,)
    best_idx = jnp.argmin(errors)
    return all_cfgs[best_idx]


# ---------------------------------------------------------------------------
# Public entry point — CUDA
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "max_iter",
        "target_joint_idx",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "eps_ori",
    ),
)
def _ls_ik_solve_cuda_jit(
    robot:             Robot,
    target_pose:       jaxlie.SE3,
    rng_key:           Array,
    previous_cfg:      Float[Array, "n_act"],
    num_seeds:         int,
    max_iter:          int,
    pos_weight:        float,
    ori_weight:        float,
    lambda_init:       float,
    eps_pos:           float,
    eps_ori:           float,
    continuity_weight: float,
    fixed_joint_mask_int: Float[Array, "n_act"],
    ancestor_mask:     Array,
    target_joint_idx:  int,
) -> Float[Array, "n_act"]:
    from ..cuda_kernels._ls_ik_cuda import ls_ik_cuda

    n_act  = robot.joints.num_actuated_joints
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    target_T = target_pose.wxyz_xyz.astype(jnp.float32)

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

    # ── CUDA LM ───────────────────────────────────────────────────────────
    cfgs, errors = ls_ik_cuda(
        seeds         = seeds[None],        # (1, n_seeds, n_act)
        twists        = robot.joints.twists,
        parent_tf     = robot.joints.parent_transforms,
        parent_idx    = robot.joints.parent_indices,
        act_idx       = robot.joints.actuated_indices,
        mimic_mul     = robot.joints.mimic_multiplier,
        mimic_off     = robot.joints.mimic_offset,
        mimic_act_idx = robot.joints.mimic_act_indices,
        topo_inv      = robot.joints._topo_sort_inv,
        ancestor_mask = ancestor_mask,
        target_T      = target_T[None],     # (1, 7)
        lower         = lower,
        upper         = upper,
        fixed_mask    = fixed_joint_mask_int,
        target_jnt    = target_joint_idx,
        max_iter      = max_iter,
        pos_weight    = pos_weight,
        ori_weight    = ori_weight,
        lambda_init   = lambda_init,
        eps_pos       = eps_pos,
        eps_ori       = eps_ori,
    )
    cfgs   = cfgs[0]    # (n_seeds, n_act)
    errors = errors[0]  # (n_seeds,)

    # ── Winner selection ───────────────────────────────────────────────────
    final_errors = (
        errors
        + continuity_weight * jnp.sum((cfgs - previous_cfg) ** 2, axis=-1)
    )
    best_idx = jnp.argmin(final_errors)
    return cfgs[best_idx]

def ls_ik_solve_cuda(
    robot:             Robot,
    target_link_index: int,
    target_pose:       jaxlie.SE3,
    rng_key:           Array,
    previous_cfg:      Float[Array, "n_act"],
    num_seeds:         int   = 32,
    max_iter:          int   = 60,
    pos_weight:        float = 50.0,
    ori_weight:        float = 10.0,
    lambda_init:       float = 5e-3,
    eps_pos:           float = 1e-8,
    eps_ori:           float = 1e-8,
    continuity_weight: float = 0.0,
    fixed_joint_mask:  Float[Array, "n_act"] | None = None,
) -> Float[Array, "n_act"]:
    """CUDA alternative to :func:`ls_ik_solve`.

    Offloads the multi-seed LM loop to a single CUDA kernel via JAX FFI.
    The kernel (``_ls_ik_cuda_kernel.cu``) is structurally equivalent to
    the JAX solver but with no Python overhead per step.

    Requires ``_ls_ik_cuda_lib.so`` compiled from ``_ls_ik_cuda_kernel.cu``:
        bash src/pyronot/cuda_kernels/build_ls_ik_cuda.sh

    Args:
        robot:              The robot model.
        target_link_index:  Index of the target link.
        target_pose:        Desired SE(3) world pose.
        rng_key:            JAX PRNG key.
        previous_cfg:       Previous joint configuration.
        num_seeds:          Number of parallel seeds.
        max_iter:           LM iteration budget per seed.
        pos_weight:         Weight on position residual.
        ori_weight:         Weight on orientation residual.
        lambda_init:        Initial LM damping factor.
        eps_pos:            Position convergence threshold [m].
        eps_ori:            Orientation convergence threshold [rad].
        continuity_weight:  Weight on ‖q − prev‖² in winner selection only.
        fixed_joint_mask:   Boolean mask; True = joint must not move.

    Returns:
        Best joint configuration found, shape ``(n_act,)``.
    """
    n_act  = robot.joints.num_actuated_joints

    if fixed_joint_mask is None:
        fixed_joint_mask_int = jnp.zeros(n_act, dtype=jnp.int32)
    else:
        fixed_joint_mask_int = fixed_joint_mask.astype(jnp.int32)

    # ── Pre-compute ancestor mask (Python level) ───────────────────────────
    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    target_joint_idx        = int(parent_joint_indices_np[target_link_index])

    parent_idx_np    = np.array(robot.joints.parent_indices)
    ancestor_mask_np = np.zeros(robot.joints.num_joints, dtype=np.int32)
    j = target_joint_idx
    while j >= 0:
        ancestor_mask_np[j] = 1
        j = int(parent_idx_np[j])
    ancestor_mask = jnp.array(ancestor_mask_np)
    return _ls_ik_solve_cuda_jit(
        robot=robot,
        target_pose=target_pose,
        rng_key=rng_key,
        previous_cfg=previous_cfg,
        num_seeds=num_seeds,
        max_iter=max_iter,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        continuity_weight=continuity_weight,
        fixed_joint_mask_int=fixed_joint_mask_int,
        ancestor_mask=ancestor_mask,
        target_joint_idx=target_joint_idx,
    )


# ---------------------------------------------------------------------------
# Public entry point — CUDA batched
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "max_iter",
        "target_joint_idx",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "eps_ori",
    ),
)
def _ls_ik_solve_cuda_batch_jit(
    robot:             Robot,
    target_poses:      jaxlie.SE3,
    rng_key:           Array,
    previous_cfgs:     Float[Array, "n_problems n_act"],
    num_seeds:         int,
    max_iter:          int,
    pos_weight:        float,
    ori_weight:        float,
    lambda_init:       float,
    eps_pos:           float,
    eps_ori:           float,
    continuity_weight: float,
    fixed_joint_mask_int: Float[Array, "n_act"],
    ancestor_mask:     Array,
    target_joint_idx:  int,
) -> Float[Array, "n_problems n_act"]:
    from ..cuda_kernels._ls_ik_cuda import ls_ik_cuda

    n_act      = robot.joints.num_actuated_joints
    lower      = robot.joints.lower_limits
    upper      = robot.joints.upper_limits
    n_problems = previous_cfgs.shape[0]

    # Batched target poses: (n_problems, 7)
    target_T_batch = target_poses.wxyz_xyz.astype(jnp.float32)  # (n_problems, 7)

    # Seed generation — vectorised across problems
    n_warm   = max(1, num_seeds // 2)
    n_random = num_seeds - n_warm

    key_warm, key_random = jax.random.split(rng_key)

    warm_seeds = jnp.clip(
        previous_cfgs[:, None, :] + jax.random.normal(key_warm, (n_problems, n_warm, n_act)) * 0.05,
        lower, upper,
    )  # (n_problems, n_warm, n_act)
    warm_seeds = jnp.where(fixed_joint_mask_int[None, None, :], previous_cfgs[:, None, :], warm_seeds)

    random_seeds = jax.random.uniform(
        key_random, (n_problems, n_random, n_act), minval=lower, maxval=upper
    )  # (n_problems, n_random, n_act)
    random_seeds = jnp.where(
        fixed_joint_mask_int[None, None, :], previous_cfgs[:, None, :], random_seeds
    )

    seeds = jnp.concatenate([warm_seeds, random_seeds], axis=1)  # (n_problems, n_seeds, n_act)

    # CUDA batched LM
    cfgs, errors = ls_ik_cuda(
        seeds         = seeds,
        twists        = robot.joints.twists,
        parent_tf     = robot.joints.parent_transforms,
        parent_idx    = robot.joints.parent_indices,
        act_idx       = robot.joints.actuated_indices,
        mimic_mul     = robot.joints.mimic_multiplier,
        mimic_off     = robot.joints.mimic_offset,
        mimic_act_idx = robot.joints.mimic_act_indices,
        topo_inv      = robot.joints._topo_sort_inv,
        ancestor_mask = ancestor_mask,
        target_T      = target_T_batch,
        lower         = lower,
        upper         = upper,
        fixed_mask    = fixed_joint_mask_int,
        target_jnt    = target_joint_idx,
        max_iter      = max_iter,
        pos_weight    = pos_weight,
        ori_weight    = ori_weight,
        lambda_init   = lambda_init,
        eps_pos       = eps_pos,
        eps_ori       = eps_ori,
    )  # cfgs: (n_problems, n_seeds, n_act), errors: (n_problems, n_seeds)

    # Winner selection per problem
    final_errors = (
        errors
        + continuity_weight * jnp.sum((cfgs - previous_cfgs[:, None, :]) ** 2, axis=-1)
    )  # (n_problems, n_seeds)
    best_idxs = jnp.argmin(final_errors, axis=1)  # (n_problems,)
    return cfgs[jnp.arange(n_problems), best_idxs]  # (n_problems, n_act)

def ls_ik_solve_cuda_batch(
    robot:             Robot,
    target_link_index: int,
    target_poses:      jaxlie.SE3,
    rng_key:           Array,
    previous_cfgs:     Float[Array, "n_problems n_act"],
    num_seeds:         int   = 32,
    max_iter:          int   = 60,
    pos_weight:        float = 50.0,
    ori_weight:        float = 10.0,
    lambda_init:       float = 5e-3,
    eps_pos:           float = 1e-8,
    eps_ori:           float = 1e-8,
    continuity_weight: float = 0.0,
    fixed_joint_mask:  Float[Array, "n_act"] | None = None,
) -> Float[Array, "n_problems n_act"]:
    """Batched CUDA LS-IK: solve n_problems targets in a single kernel launch.

    Args:
        robot:              The robot model.
        target_link_index:  Index of the target link in ``robot.links.names``.
        target_poses:       Batch of SE(3) targets, shape ``(n_problems,)`` pytree.
        rng_key:            JAX PRNG key (split internally per problem).
        previous_cfgs:      Previous configurations, shape ``(n_problems, n_act)``.
        num_seeds:          Number of parallel seeds per problem.
        max_iter:           LM iteration budget per seed.
        pos_weight:         Weight on position residual.
        ori_weight:         Weight on orientation residual.
        lambda_init:        Initial LM damping factor.
        eps_pos:            Position convergence threshold [m].
        eps_ori:            Orientation convergence threshold [rad].
        continuity_weight:  Weight on ‖q − prev‖² in winner selection only.
        fixed_joint_mask:   Boolean mask; True = joint must not move.

    Returns:
        Best joint configurations, shape ``(n_problems, n_act)``.
    """
    n_act      = robot.joints.num_actuated_joints

    if fixed_joint_mask is None:
        fixed_joint_mask_int = jnp.zeros(n_act, dtype=jnp.int32)
    else:
        fixed_joint_mask_int = fixed_joint_mask.astype(jnp.int32)

    # Ancestor mask (same for all problems — same target link)
    parent_joint_indices_np = np.array(robot.links.parent_joint_indices)
    target_joint_idx        = int(parent_joint_indices_np[target_link_index])
    parent_idx_np    = np.array(robot.joints.parent_indices)
    ancestor_mask_np = np.zeros(robot.joints.num_joints, dtype=np.int32)
    j = target_joint_idx
    while j >= 0:
        ancestor_mask_np[j] = 1
        j = int(parent_idx_np[j])
    ancestor_mask = jnp.array(ancestor_mask_np)
    return _ls_ik_solve_cuda_batch_jit(
        robot=robot,
        target_poses=target_poses,
        rng_key=rng_key,
        previous_cfgs=previous_cfgs,
        num_seeds=num_seeds,
        max_iter=max_iter,
        pos_weight=pos_weight,
        ori_weight=ori_weight,
        lambda_init=lambda_init,
        eps_pos=eps_pos,
        eps_ori=eps_ori,
        continuity_weight=continuity_weight,
        fixed_joint_mask_int=fixed_joint_mask_int,
        ancestor_mask=ancestor_mask,
        target_joint_idx=target_joint_idx,
    )
