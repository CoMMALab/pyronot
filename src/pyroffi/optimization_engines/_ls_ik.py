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
from ..collision._cuda_collision import _extract_world_arrays


# Module-level cache keyed on object identity.  All inputs are static Python
# objects whose content never changes between calls, so identity equality is
# correct.  Caching makes _prepare_ls_collision_buffers safe to call inside
# jax.lax.scan bodies: the cache hit is a pure Python dict lookup (no JAX
# operations), so the abstract-tracing context that scan establishes cannot
# infect any array accesses inside the function body.
_COLLISION_BUFFER_CACHE: dict = {}


def _prepare_ls_collision_buffers(
    robot: Robot,
    collision_checker: Any | None,
    world_geom: Any | None,
) -> tuple[
    Float[Array, "n_rs 4"],
    jnp.ndarray,
    Float[Array, "n_ws 4"],
    Float[Array, "n_wc 7"],
    Float[Array, "n_wb 15"],
    Float[Array, "n_wh 6"],
    jnp.ndarray,
    jnp.ndarray,
    bool,
]:
    geom_key = (
        tuple(id(g) for g in world_geom)
        if isinstance(world_geom, (list, tuple))
        else id(world_geom)
    )
    cache_key = (id(robot), id(collision_checker), geom_key)
    if cache_key in _COLLISION_BUFFER_CACHE:
        return _COLLISION_BUFFER_CACHE[cache_key]
    result = _prepare_ls_collision_buffers_uncached(robot, collision_checker, world_geom)
    _COLLISION_BUFFER_CACHE[cache_key] = result
    return result


def _prepare_ls_collision_buffers_uncached(
    robot: Robot,
    collision_checker: Any | None,
    world_geom: Any | None,
) -> tuple[
    Float[Array, "n_rs 4"],
    jnp.ndarray,
    Float[Array, "n_ws 4"],
    Float[Array, "n_wc 7"],
    Float[Array, "n_wb 15"],
    Float[Array, "n_wh 6"],
    jnp.ndarray,
    jnp.ndarray,
    bool,
]:
    empty_rs = jnp.zeros((0, 4), dtype=jnp.float32)
    empty_idx = jnp.zeros((0,), dtype=jnp.int32)
    empty_ws = jnp.zeros((0, 4), dtype=jnp.float32)
    empty_wc = jnp.zeros((0, 7), dtype=jnp.float32)
    empty_wb = jnp.zeros((0, 15), dtype=jnp.float32)
    empty_wh = jnp.zeros((0, 6), dtype=jnp.float32)
    empty_pair = jnp.zeros((0,), dtype=jnp.int32)

    if collision_checker is None or world_geom is None:
        return empty_rs, empty_idx, empty_ws, empty_wc, empty_wb, empty_wh, empty_pair, empty_pair, False

    inner = getattr(collision_checker, "_inner", collision_checker)
    coll = getattr(inner, "coll", None)
    if coll is None or not hasattr(coll, "pose") or not hasattr(coll, "radius"):
        return empty_rs, empty_idx, empty_ws, empty_wc, empty_wb, empty_wh, empty_pair, empty_pair, False

    # Use np.asarray on the raw wxyz_xyz array and slice in numpy — calling
    # coll.pose.translation() would invoke a JAX slice operation that creates
    # an abstract traced value inside jax.lax.scan bodies.
    centers = np.asarray(coll.pose.wxyz_xyz)[..., 4:7].astype(np.float32)
    radii = np.asarray(coll.radius, dtype=np.float32)

    rs_list: list[np.ndarray] = []
    jidx_list: list[int] = []
    # link_idx_per_sphere[flat_sphere_idx] = source link index. Used to expand
    # link-level active_idx_i/_j pairs into flat sphere-index pairs for the
    # CUDA kernel's self-collision pass.
    link_per_sphere: list[int] = []
    parent_joint = np.asarray(robot.links.parent_joint_indices, dtype=np.int32)

    # Spherized model: centers shape (N, S, 3), radii shape (N, S).
    if centers.ndim == 3 and radii.ndim == 2:
        n_links, n_spheres, _ = centers.shape
        if n_links > parent_joint.shape[0]:
            n_links = parent_joint.shape[0]
        for li in range(n_links):
            jidx = int(parent_joint[li])
            if jidx < 0:
                continue
            for si in range(n_spheres):
                r = float(radii[li, si])
                if r <= 0.0 or r < -1e8:
                    continue
                c = centers[li, si]
                rs_list.append(np.array([c[0], c[1], c[2], r], dtype=np.float32))
                jidx_list.append(jidx)
                link_per_sphere.append(li)
    # Capsule model: centers shape (N, 3), radii/height shape (N,).
    elif centers.ndim == 2 and radii.ndim == 1 and hasattr(coll, "height") and hasattr(coll, "axis"):
        axes = np.asarray(coll.axis, dtype=np.float32)
        heights = np.asarray(coll.height, dtype=np.float32)
        n_links = min(centers.shape[0], parent_joint.shape[0])
        for li in range(n_links):
            jidx = int(parent_joint[li])
            if jidx < 0:
                continue
            r = float(radii[li])
            h = float(heights[li])
            if r <= 0.0 or r < -1e8:
                continue
            c = centers[li]
            a = axes[li]
            off = 0.5 * h * a
            c0 = c - off
            c1 = c + off
            rs_list.append(np.array([c0[0], c0[1], c0[2], r], dtype=np.float32))
            rs_list.append(np.array([c1[0], c1[1], c1[2], r], dtype=np.float32))
            jidx_list.extend([jidx, jidx])
            link_per_sphere.extend([li, li])
    else:
        return empty_rs, empty_idx, empty_ws, empty_wc, empty_wb, empty_wh, empty_pair, empty_pair, False

    if len(rs_list) == 0:
        return empty_rs, empty_idx, empty_ws, empty_wc, empty_wb, empty_wh, empty_pair, empty_pair, False

    # Expand active_idx (link-level pairs) into flat (sphere_a, sphere_b)
    # pairs. The kernel sums (d - margin)^2 over each pair, so duplicating per
    # sphere within a link pair is a stronger barrier than the JAX min-distance
    # but still pulls toward the same constraint manifold.
    active_i = getattr(inner, "active_idx_i", None)
    active_j = getattr(inner, "active_idx_j", None)
    self_pair_i_list: list[int] = []
    self_pair_j_list: list[int] = []
    if active_i is not None and active_j is not None:
        active_i_np = np.asarray(active_i, dtype=np.int32)
        active_j_np = np.asarray(active_j, dtype=np.int32)
        link_per_sphere_np = np.asarray(link_per_sphere, dtype=np.int32)
        # Bucket flat sphere indices by their link.
        spheres_by_link: dict[int, list[int]] = {}
        for flat_idx, li in enumerate(link_per_sphere_np.tolist()):
            spheres_by_link.setdefault(li, []).append(flat_idx)
        for li, lj in zip(active_i_np.tolist(), active_j_np.tolist()):
            for sa in spheres_by_link.get(int(li), ()):
                for sb in spheres_by_link.get(int(lj), ()):
                    self_pair_i_list.append(sa)
                    self_pair_j_list.append(sb)

    self_pair_i_arr = (
        jnp.array(np.array(self_pair_i_list, dtype=np.int32), dtype=jnp.int32)
        if self_pair_i_list else empty_pair
    )
    self_pair_j_arr = (
        jnp.array(np.array(self_pair_j_list, dtype=np.int32), dtype=jnp.int32)
        if self_pair_j_list else empty_pair
    )

    world_items = world_geom if isinstance(world_geom, (list, tuple)) else (world_geom,)
    ws_chunks: list[np.ndarray] = []
    wc_chunks: list[np.ndarray] = []
    wb_chunks: list[np.ndarray] = []
    wh_chunks: list[np.ndarray] = []
    for wg in world_items:
        ws_np_i, wc_np_i, wb_np_i, wh_np_i = _extract_world_arrays(wg)
        if ws_np_i.shape[0] > 0:
            ws_chunks.append(ws_np_i)
        if wc_np_i.shape[0] > 0:
            wc_chunks.append(wc_np_i)
        if wb_np_i.shape[0] > 0:
            wb_chunks.append(wb_np_i)
        if wh_np_i.shape[0] > 0:
            wh_chunks.append(wh_np_i)

    ws_np = np.concatenate(ws_chunks, axis=0) if ws_chunks else np.zeros((0, 4), dtype=np.float32)
    wc_np = np.concatenate(wc_chunks, axis=0) if wc_chunks else np.zeros((0, 7), dtype=np.float32)
    wb_np = np.concatenate(wb_chunks, axis=0) if wb_chunks else np.zeros((0, 15), dtype=np.float32)
    wh_np = np.concatenate(wh_chunks, axis=0) if wh_chunks else np.zeros((0, 6), dtype=np.float32)
    return (
        jnp.array(np.stack(rs_list, axis=0), dtype=jnp.float32),
        jnp.array(np.array(jidx_list, dtype=np.int32), dtype=jnp.int32),
        jnp.array(ws_np, dtype=jnp.float32),
        jnp.array(wc_np, dtype=jnp.float32),
        jnp.array(wb_np, dtype=jnp.float32),
        jnp.array(wh_np, dtype=jnp.float32),
        self_pair_i_arr,
        self_pair_j_arr,
        True,
    )

# ---------------------------------------------------------------------------
# Single-seed LM solve
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("target_link_indices", "max_iter", "constraint_fns"),
)
def _ls_ik_single(
    cfg:                   Float[Array, "n_act"],
    robot:                 Robot,
    target_link_indices:   tuple[int, ...],
    target_poses:          tuple,
    max_iter:              int,
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
    """Levenberg-Marquardt refinement from a single starting configuration.

    Normal equations
        (Js^T Js + lambda * I) p = -Js^T fw
    where Js is the column-scaled weighted Jacobian and fw = W * f.

    The residual vector is:
        [W*f_0 | W*f_1 | ... | W*f_{N-1} | sqrt_wc*f_coll]
    where each f_i is the 6-vector SE(3) log-map residual for EE i.

    The trust-region radius uses the FIRST EE's pos/ori error (primary EE
    drives the trust region), not the combined error.

    FK fusion
        Task and constraint residuals are wrapped in a single
        ``fused_scaled(q)`` callable before the ``jax.vjp`` call.
        XLA CSE then deduplicates the ``robot.forward_kinematics(q)`` call
        shared between all ``_ik_residual`` calls and the constraint
        functions, replacing many FK forward passes with one.  The
        line-search also reuses ``fused_scaled``, halving the number of FK
        evaluations per LM step.

    Normal equations
        Solved in float32.  Jacobi column scaling provides sufficient
        conditioning for real-time IK without the float64 upcast.

    Line search
        Five step-size candidates {1, 0.5, 0.25, 0.1, 0.025} are evaluated
        in parallel via vmap; the one producing the lowest weighted residual
        is accepted.
    """
    n     = cfg.shape[0]
    n_c   = len(constraint_fns)
    n_ee  = len(target_link_indices)
    W     = jnp.concatenate([
        jnp.full(3, pos_weight, dtype=cfg.dtype),
        jnp.full(3, ori_weight, dtype=cfg.dtype),
    ])  # (6,)

    # ── Build residual callable once, outside the scan ─────────────────────
    # fused_scaled returns [W*f_0 | ... | W*f_{N-1} | sqrt_wc*f_coll]
    # XLA CSE collapses all robot.forward_kinematics(q) calls into one.
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

    def lm_step(carry, _):
        c, lam, best_c, best_err = carry

        # ── Jacobian + weighted residual (single FK forward pass) ────────
        f_all, vjp_fn = jax.vjp(fused_scaled, c)
        J_all = jax.vmap(lambda g: vjp_fn(g)[0])(
            jnp.eye(n_out, dtype=f_all.dtype)
        )  # (n_out, n)
        fw_eff  = f_all          # already weighted
        Jw_eff  = J_all
        curr_err = jnp.dot(fw_eff, fw_eff)

        # ── Jacobi column scaling ────────────────────────────────────────
        col_scale = jnp.linalg.norm(Jw_eff, axis=0) + 1e-8   # (n,)
        Js        = Jw_eff / col_scale[None, :]               # unit-column Jacobian

        # ── Normal equations with LM damping (float32) ──────────────────
        A   = Js.T @ Js + lam * jnp.eye(n, dtype=Js.dtype)
        rhs = -(Js.T @ fw_eff)
        p   = jnp.linalg.solve(A, rhs)   # (n,) in scaled space

        # Unscale and freeze fixed joints.
        delta = p / col_scale
        delta = jnp.where(fixed_joint_mask, 0.0, delta)

        # ── Trust-region step clipping ───────────────────────────────────
        # Use the MAX error across all EEs so the trust region stays open
        # as long as any end-effector still needs large steps.
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

    # ── Initial error ──────────────────────────────────────────────────────
    init_f_all = fused_scaled(cfg)
    init_err   = jnp.dot(init_f_all, init_f_all)

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
    static_argnames=("target_link_indices", "num_seeds", "max_iter", "constraint_fns"),
)
def ls_ik_solve(
    robot:              Robot,
    target_link_indices: tuple[int, ...],
    target_poses:       tuple,
    rng_key:            Array,
    previous_cfg:       Float[Array, "n_act"],
    num_seeds:          int   = 32,
    max_iter:           int   = 60,
    pos_weight:         float = 50.0,
    ori_weight:         float = 10.0,
    lambda_init:        float = 5e-3,
    continuity_weight:  float = 0.0,
    fixed_joint_mask:   Float[Array, "n_act"] | None = None,
    constraint_fns:     tuple = (),
    constraint_args:    tuple = (),
    constraint_weights: Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    """Solve IK via multi-seed Levenberg-Marquardt (no coarse phase).

    All ``num_seeds`` seeds are refined directly with LM — there is no
    Hamiltonian coordinate-descent coarse phase.  This makes the solver
    a clean Gauss-Newton least-squares method suitable for comparison with
    HJCD-IK.

    Multi-EE support
        Pass multiple end-effectors via ``target_link_indices`` and
        ``target_poses`` tuples.  The residual vector becomes
        ``[W*f_0 | W*f_1 | ... | W*f_{N-1} | sqrt_wc*f_coll]`` where each
        ``f_i`` is the 6-vector SE(3) log-map residual for EE ``i``.
        For a single EE, pass ``target_link_indices=(idx,)`` and
        ``target_poses=(pose,)``.

    Seed generation
        Half the seeds are warm-starts near ``previous_cfg`` (σ = 0.05).
        The remaining half are drawn uniformly across joint limits.

    Per-seed solve
        Each seed gets ``max_iter`` LM iterations with Jacobi column
        scaling, a 5-point vectorised line search, and best-config tracking.
        All seeds run in parallel via ``jax.vmap``.

    Kinematic constraints
        Optional penalty terms added to the LM cost function.  Each constraint
        contributes ``weight_i * c_i(q)^2`` to the minimised objective so the
        solver simultaneously satisfies the task and the constraints.  Constraint
        penalties are also included in the winner selection score.

        Constraint functions must have signature
            ``c(cfg: Array, robot: Robot) -> Array``  (scalar)
        and should return 0 when satisfied, positive when violated.  Equality
        constraints (e.g. keeping the EE on a plane) may return signed values;
        both signs are penalised symmetrically.

    Winner selection
        Same as ``hjcd_solve``: lowest weighted task-space residual (summed
        over all EEs) plus optional ``continuity_weight · ‖q − prev‖²``
        tie-breaker, and constraint penalties when applicable.

    Args:
        robot:               The robot model.
        target_link_indices: Tuple of target link indices (static, for JIT).
                             Use a 1-tuple for single-EE problems.
        target_poses:        Tuple of desired SE(3) world poses (dynamic pytree).
        rng_key:             JAX PRNG key.
        previous_cfg:        Previous joint configuration for warm-starting
                             and continuity-aware winner selection.
        num_seeds:           Number of parallel seeds (default 32).
        max_iter:            LM iterations per seed (default 60).
        pos_weight:          Weight on position residual (default 50.0).
        ori_weight:          Weight on orientation residual (default 10.0).
        lambda_init:         Initial LM damping factor (default 5e-3).
        continuity_weight:   Weight on ‖q − prev‖² in winner selection only.
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

    # ── Seed generation ────────────────────────────────────────────────────
    n_warm   = max(1, num_seeds // 2)
    n_random = num_seeds - n_warm

    key_warm, key_random = jax.random.split(rng_key)

    # For multi-EE problems split warm seeds into a tight (σ=0.05) and a
    # loose (σ=0.3) half.  The tight half provides continuity; the loose
    # half allows escape from configurations where one EE is converged but
    # another is stuck in a local basin near the warm-start.
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

    # ── Multi-seed LM (parallel over seeds) ───────────────────────────────
    all_cfgs = jax.vmap(
        lambda cfg: _ls_ik_single(
            cfg, robot, target_link_indices, target_poses,
            max_iter, lambda_init, pos_weight, ori_weight,
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
# Public entry point — CUDA
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=(
        "num_seeds",
        "max_iter",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "eps_ori",
        "enable_collision",
        "collision_weight",
        "collision_margin",
        "constraint_fns",
        "target_link_indices",
    ),
)
def _ls_ik_solve_cuda_jit(
    robot:                Robot,
    target_poses:         tuple,
    rng_key:              Array,
    previous_cfg:         Float[Array, "n_act"],
    num_seeds:            int,
    max_iter:             int,
    pos_weight:           float,
    ori_weight:           float,
    lambda_init:          float,
    eps_pos:              float,
    eps_ori:              float,
    continuity_weight:    float,
    fixed_joint_mask_int: Float[Array, "n_act"],
    ancestor_masks:       Array,
    target_jnts:          Array,
    robot_spheres_local:  Float[Array, "n_rs 4"],
    robot_sphere_joint_idx: jnp.ndarray,
    world_spheres:        Float[Array, "n_ws 4"],
    world_capsules:       Float[Array, "n_wc 7"],
    world_boxes:          Float[Array, "n_wb 15"],
    world_halfspaces:     Float[Array, "n_wh 6"],
    self_pair_i:          Array,
    self_pair_j:          Array,
    enable_collision:     bool,
    collision_weight:     float,
    collision_margin:     float,
    target_link_indices:  tuple[int, ...],
    constraint_fns:       tuple = (),
    constraint_args:      tuple = (),
    constraint_weights:   Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_act"]:
    from ..cuda_kernels._ls_ik_cuda import ls_ik_cuda

    n_act  = robot.joints.num_actuated_joints
    lower  = robot.joints.lower_limits
    upper  = robot.joints.upper_limits

    # Stack all EE target poses: (n_ee, 7).
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

    # ── CUDA LM (all EEs simultaneously) ─────────────────────────────────
    cfgs, errors = ls_ik_cuda(
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
        self_pair_i    = self_pair_i,
        self_pair_j    = self_pair_j,
        lower          = lower,
        upper          = upper,
        fixed_mask     = fixed_joint_mask_int,
        max_iter       = max_iter,
        pos_weight     = pos_weight,
        ori_weight     = ori_weight,
        lambda_init    = lambda_init,
        eps_pos        = eps_pos,
        eps_ori        = eps_ori,
        enable_collision = enable_collision,
        collision_weight = collision_weight,
        collision_margin = collision_margin,
    )
    cfgs   = cfgs[0]    # (n_seeds, n_act)
    errors = errors[0]  # (n_seeds,) — all EE weighted errors from CUDA

    # CUDA now covers all EEs — no rescoring needed.
    base_errors = errors

    if len(constraint_fns) > 0:
        def constraint_penalty(cfg):
            c_vals = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(len(constraint_fns))])
            return jnp.sum(constraint_weights * c_vals ** 2)

        constraint_errors = jax.vmap(constraint_penalty)(cfgs)  # (n_seeds,)
        final_errors = (
            base_errors
            + constraint_errors
            + continuity_weight * jnp.sum((cfgs - previous_cfg) ** 2, axis=-1)
        )
    else:
        constraint_errors = jnp.zeros(cfgs.shape[0])
        final_errors = (
            base_errors
            + continuity_weight * jnp.sum((cfgs - previous_cfg) ** 2, axis=-1)
        )

    best_idx = jnp.argmin(final_errors)
    # Return winner + its constraint cost (already computed — avoids a redundant
    # FK evaluation in the Python-level early-exit check).
    if len(constraint_fns) > 0:
        return cfgs[best_idx], constraint_errors[best_idx]
    return cfgs[best_idx], jnp.zeros(())


def ls_ik_solve_cuda(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3 | tuple,
    rng_key:             Array,
    previous_cfg:        Float[Array, "n_act"],
    num_seeds:           int   = 32,
    max_iter:            int   = 60,
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
    collision_constraint_indices: Sequence[int] | None = None,
    collision_free:      bool = False,
    collision_checker:   Any | None = None,
    collision_world:     Any | None = None,
    collision_weight:    float = 1e4,
    collision_margin:    float = 0.02,
    constraint_refine_iters: int = 12,
) -> Float[Array, "n_act"]:
    """CUDA alternative to :func:`ls_ik_solve`.

    Offloads the multi-seed LM loop to a single CUDA kernel via JAX FFI.
    The kernel (``_ls_ik_cuda_kernel.cu``) is structurally equivalent to
    the JAX solver but with no Python overhead per step.

    Requires ``_ls_ik_cuda_lib.so`` compiled from ``_ls_ik_cuda_kernel.cu``:
        bash src/pyroffi/cuda_kernels/build_ls_ik_cuda.sh

    Multi-EE support
        Pass multiple end-effectors via ``target_link_indices`` (tuple) and
        ``target_poses`` (tuple).  The CUDA kernel optimises only the FIRST
        EE; remaining EEs are incorporated into winner selection and
        post-CUDA JAX refinement.  For a single EE, pass
        ``target_link_indices=(idx,)`` and ``target_poses=(pose,)``.

    Kinematic constraints
        Because the CUDA kernel cannot call arbitrary Python/JAX functions,
        constraints are incorporated in two stages:

        1. **Winner selection** — constraint penalties are evaluated via JAX on
           all CUDA-returned candidates and added to the selection score.
        2. **Post-CUDA JAX refinement** — ``constraint_refine_iters`` LM steps
           are run on the selected winner using the full constraint-augmented JAX
           solver (:func:`_ls_ik_single`).  Set ``constraint_refine_iters=0`` to
           skip this pass and use winner selection only.  Refinement is skipped
           automatically when the CUDA winner is already constraint-feasible
           (total weighted constraint cost ≤ 1e-6).

    Args:
        robot:                   The robot model.
        target_link_indices:     Index (or tuple of indices) of target link(s).
        target_poses:            Desired SE(3) world pose (or tuple of poses).
        rng_key:                 JAX PRNG key.
        previous_cfg:            Previous joint configuration.
        num_seeds:               Number of parallel seeds.
        max_iter:                LM iteration budget per seed.
        pos_weight:              Weight on position residual.
        ori_weight:              Weight on orientation residual.
        lambda_init:             Initial LM damping factor.
        eps_pos:                 Position convergence threshold [m].
        eps_ori:                 Orientation convergence threshold [rad].
        continuity_weight:       Weight on ‖q − prev‖² in winner selection only.
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
        self_pair_i,
        self_pair_j,
        kernel_collision_enabled,
    ) = _prepare_ls_collision_buffers(robot, collision_checker, collision_world)
    kernel_collision_enabled = bool(collision_free and kernel_collision_enabled)

    winner, winner_coll_cost = _ls_ik_solve_cuda_jit(
        robot=robot,
        target_poses=target_poses_t,
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
        ancestor_masks=ancestor_masks,
        target_jnts=target_jnts,
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        self_pair_i=self_pair_i,
        self_pair_j=self_pair_j,
        enable_collision=kernel_collision_enabled,
        collision_weight=collision_weight,
        collision_margin=collision_margin,
        target_link_indices=target_link_indices,
        constraint_fns=cuda_constraint_fns,
        constraint_args=cuda_constraint_args,
        constraint_weights=cuda_constraint_weights,
    )

    # ── Post-CUDA JAX refinement with constraints only ────────────────────
    # Multi-EE is now handled in CUDA; only constraints require JAX refinement.
    needs_refinement = bool(post_constraint_fns)
    if needs_refinement and constraint_refine_iters > 0:
        fmask = (
            fixed_joint_mask.astype(jnp.bool_)
            if fixed_joint_mask is not None
            else jnp.zeros(n_act, dtype=jnp.bool_)
        )
        winner = _ls_ik_single(
            winner, robot, target_link_indices, target_poses_t,
            constraint_refine_iters, lambda_init, pos_weight, ori_weight,
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
        "max_iter",
        "pos_weight",
        "ori_weight",
        "lambda_init",
        "eps_pos",
        "eps_ori",
        "enable_collision",
        "collision_weight",
        "collision_margin",
        "constraint_fns",
        "target_link_indices",
    ),
)
def _ls_ik_solve_cuda_batch_jit(
    robot:                Robot,
    target_poses_batch:   jaxlie.SE3,
    rng_key:              Array,
    previous_cfgs:        Float[Array, "n_problems n_act"],
    num_seeds:            int,
    max_iter:             int,
    pos_weight:           float,
    ori_weight:           float,
    lambda_init:          float,
    eps_pos:              float,
    eps_ori:              float,
    continuity_weight:    float,
    fixed_joint_mask_int: Float[Array, "n_act"],
    ancestor_masks:       Array,
    target_jnts:          Array,
    robot_spheres_local:  Float[Array, "n_rs 4"],
    robot_sphere_joint_idx: jnp.ndarray,
    world_spheres:        Float[Array, "n_ws 4"],
    world_capsules:       Float[Array, "n_wc 7"],
    world_boxes:          Float[Array, "n_wb 15"],
    world_halfspaces:     Float[Array, "n_wh 6"],
    self_pair_i:          Array,
    self_pair_j:          Array,
    enable_collision:     bool,
    collision_weight:     float,
    collision_margin:     float,
    target_link_indices:  tuple[int, ...],
    constraint_fns:       tuple = (),
    constraint_args:      tuple = (),
    constraint_weights:   Float[Array, "n_constraints"] | None = None,
) -> Float[Array, "n_problems n_act"]:
    from ..cuda_kernels._ls_ik_cuda import ls_ik_cuda

    n_act      = robot.joints.num_actuated_joints
    lower      = robot.joints.lower_limits
    upper      = robot.joints.upper_limits
    n_problems = previous_cfgs.shape[0]

    # Batched target poses: (n_problems, 7) wrapped to (n_problems, 1, 7) for n_ee=1.
    target_T_batch = target_poses_batch.wxyz_xyz.astype(jnp.float32)[:, None, :]  # (n_problems, 1, 7)

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

    # CUDA batched LM — all EEs (n_ee=1 for batch path)
    cfgs, errors = ls_ik_cuda(
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
        self_pair_i    = self_pair_i,
        self_pair_j    = self_pair_j,
        lower          = lower,
        upper          = upper,
        fixed_mask     = fixed_joint_mask_int,
        max_iter       = max_iter,
        pos_weight     = pos_weight,
        ori_weight     = ori_weight,
        lambda_init    = lambda_init,
        eps_pos        = eps_pos,
        eps_ori        = eps_ori,
        enable_collision = enable_collision,
        collision_weight = collision_weight,
        collision_margin = collision_margin,
    )  # cfgs: (n_problems, n_seeds, n_act), errors: (n_problems, n_seeds)

    # ── Winner selection per problem: task (all EEs) + constraint penalties + continuity
    if len(constraint_fns) > 0:
        # cfgs: (n_problems, n_seeds, n_act)  — flatten to (n_problems*n_seeds, n_act),
        # evaluate constraints, then reshape back.
        flat_cfgs = cfgs.reshape(n_problems * num_seeds, n_act)

        def constraint_penalty(cfg):
            c_vals = jnp.stack([constraint_fns[i](cfg, robot, constraint_args[i]) for i in range(len(constraint_fns))])
            return jnp.sum(constraint_weights * c_vals ** 2)

        flat_cpen = jax.vmap(constraint_penalty)(flat_cfgs)          # (n_problems*n_seeds,)
        cpen      = flat_cpen.reshape(n_problems, num_seeds)          # (n_problems, n_seeds)

        final_errors = (
            errors
            + cpen
            + continuity_weight * jnp.sum((cfgs - previous_cfgs[:, None, :]) ** 2, axis=-1)
        )
    else:
        final_errors = (
            errors
            + continuity_weight * jnp.sum((cfgs - previous_cfgs[:, None, :]) ** 2, axis=-1)
        )  # (n_problems, n_seeds)

    best_idxs = jnp.argmin(final_errors, axis=1)  # (n_problems,)
    return cfgs[jnp.arange(n_problems), best_idxs]  # (n_problems, n_act)


def ls_ik_solve_cuda_batch(
    robot:               Robot,
    target_link_indices: int | tuple[int, ...],
    target_poses:        jaxlie.SE3,
    rng_key:             Array,
    previous_cfgs:       Float[Array, "n_problems n_act"],
    num_seeds:           int   = 32,
    max_iter:            int   = 60,
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
    collision_constraint_indices: Sequence[int] | None = None,
    collision_free:      bool = False,
    collision_checker:   Any | None = None,
    collision_world:     Any | None = None,
    collision_weight:    float = 1e4,
    collision_margin:    float = 0.02,
    constraint_refine_iters: int = 12,
    num_solutions:       int = 1,
) -> Float[Array, "n_problems n_act"]:
    """Batched CUDA LS-IK: solve n_problems targets in a single kernel launch.

    The CUDA kernel optimises only the FIRST EE.  Remaining EEs are
    incorporated into post-CUDA JAX refinement.

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
        rng_key:                 JAX PRNG key (split internally per problem).
        previous_cfgs:           Previous configurations, shape ``(n_problems, n_act)``.
        num_seeds:               Number of parallel seeds per problem.
        max_iter:                LM iteration budget per seed.
        pos_weight:              Weight on position residual.
        ori_weight:              Weight on orientation residual.
        lambda_init:             Initial LM damping factor.
        eps_pos:                 Position convergence threshold [m].
        eps_ori:                 Orientation convergence threshold [rad].
        continuity_weight:       Weight on ‖q − prev‖² in winner selection only.
        fixed_joint_mask:        Boolean mask; True = joint must not move.
        constraints:             List of constraint callables
                                 ``c(cfg, robot) -> scalar``.
        constraint_weights:      Scalar weight for each constraint.
        constraint_refine_iters: JAX LM iterations applied post-CUDA on each
                                 problem's winner (default 12, set 0 to disable).

    Returns:
        Best joint configurations, shape ``(n_problems, n_act)``.
    """
    # Normalise scalar → 1-tuple API.
    if isinstance(target_link_indices, int):
        target_link_indices = (target_link_indices,)

    n_act      = robot.joints.num_actuated_joints
    num_solutions = max(1, int(num_solutions))
    n_user_problems = previous_cfgs.shape[0]

    if num_solutions > 1:
        target_poses = jaxlie.SE3(
            jnp.repeat(target_poses.wxyz_xyz, num_solutions, axis=0)
        )
        previous_cfgs = jnp.repeat(previous_cfgs, num_solutions, axis=0)
        rng_key = jax.random.fold_in(rng_key, num_solutions)

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
    target_joint_idx        = int(parent_joint_indices_np[target_link_indices[0]])
    parent_idx_np    = np.array(robot.joints.parent_indices)
    n_joints         = robot.joints.num_joints
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
        self_pair_i,
        self_pair_j,
        kernel_collision_enabled,
    ) = _prepare_ls_collision_buffers(robot, collision_checker, collision_world)
    kernel_collision_enabled = bool(collision_free and kernel_collision_enabled)

    winners = _ls_ik_solve_cuda_batch_jit(
        robot=robot,
        target_poses_batch=target_poses,
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
        ancestor_masks=ancestor_masks,
        target_jnts=target_jnts,
        robot_spheres_local=robot_spheres_local,
        robot_sphere_joint_idx=robot_sphere_joint_idx,
        world_spheres=world_spheres,
        world_capsules=world_capsules,
        world_boxes=world_boxes,
        world_halfspaces=world_halfspaces,
        self_pair_i=self_pair_i,
        self_pair_j=self_pair_j,
        enable_collision=kernel_collision_enabled,
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

        # For batched, target_poses is the batch for the first EE only.
        # Wrap each problem's first-EE pose as a 1-tuple for the single-EE
        # or multi-EE call.  Note: only single-EE is supported for the batch
        # path's post-CUDA refinement (multi-EE batch is handled by winner selection).
        winners = jax.vmap(
            lambda cfg, wxyz_xyz: _ls_ik_single(
                cfg, robot, target_link_indices,
                (jaxlie.SE3(wxyz_xyz.astype(cfg.dtype)),),
                constraint_refine_iters, lambda_init, pos_weight, ori_weight,
                lower, upper, fmask,
                constraint_fns=post_constraint_fns,
                constraint_args=post_constraint_args,
                constraint_weights=post_constraint_weights,
            )
        )(winners, target_poses.wxyz_xyz)

    if num_solutions > 1:
        winners = winners.reshape(n_user_problems, num_solutions, n_act)
    return winners
