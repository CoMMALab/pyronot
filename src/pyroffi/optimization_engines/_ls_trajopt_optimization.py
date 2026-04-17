"""Least-Squares Trajectory Optimization.

This module formulates trajectory optimization as one large least-squares
problem over the full flattened trajectory state x = vec(q_{0:T-1}).

Design goals:
- Keep the public API aligned with existing trajopt engines.
- Reuse SCO/CHOMP collision linearization ideas.
- Solve each outer linearized subproblem with LM/Gauss-Newton on the stacked
  residual vector.
- Optionally reuse existing LS-IK solvers (JAX or CUDA) as a post-step
  projection stage for a tracked end-effector path.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ..collision import RobotCollision, colldist_from_sdf
from ._ik_primitives import _LS_ALPHAS
from ._ls_ik import ls_ik_solve, ls_ik_solve_cuda_batch


_TRAJOPT_LS_ALPHAS = _LS_ALPHAS


@dataclass(frozen=True)
class LsTrajOptConfig:
    """Hyper-parameters for least-squares trajectory optimization."""

    n_outer_iters: int = 8
    """Number of collision-linearization outer iterations."""

    n_ls_iters: int = 8
    """Levenberg-Marquardt iterations per outer step."""

    n_cg_iters: int = 10
    """Conjugate-gradient iterations used by matrix-free normal-equation solves."""

    cg_tol: float = 1e-4
    """Relative tolerance for the matrix-free CG solve."""

    inner_stall_patience: int = 2
    """Terminate effective LM work early after this many non-improving steps."""

    lambda_init: float = 5e-3
    """Initial LM damping."""

    lambda_min: float = 1e-8
    """Lower clamp for LM damping."""

    lambda_max: float = 1e6
    """Upper clamp for LM damping."""

    w_smooth: float = 1.0
    """Global smoothness multiplier."""

    w_acc: float = 0.6
    """Acceleration residual weight inside smoothness."""

    w_jerk: float = 0.2
    """Jerk residual weight inside smoothness."""

    w_collision: float = 1.0
    """Initial collision penalty weight."""

    w_collision_max: float = 100.0
    """Maximum collision penalty weight."""

    penalty_scale: float = 3.0
    """Per-outer-step multiplier for collision penalty."""

    collision_margin: float = 0.01
    """Collision activation margin in meters."""

    w_limits: float = 1.0
    """Soft joint-limit residual weight."""

    w_trust: float = 0.5
    """Trust residual weight toward the linearization point."""

    w_endpoint: float = 100.0
    """Endpoint pinning residual weight."""

    smooth_min_temperature: float = 0.05
    """Temperature for per-group smooth-min collision reduction."""

    max_delta_per_step: float = 0.1
    """Per-joint clamp for LM step updates."""

    # Optional LS-IK projection stage to leverage existing IK solvers.
    ik_projection_link_index: int | None = None
    """Tracked link index for post-optimization LS-IK projection."""

    ik_projection_num_seeds: int = 8
    """Seed count used by LS-IK projection."""

    ik_projection_max_iter: int = 20
    """LM iterations used by LS-IK projection."""

    ik_projection_continuity_weight: float = 0.1
    """Continuity tie-breaker in LS-IK projection."""

    use_legacy_cuda_kernel: bool = False
    """Opt into the legacy custom CUDA LS kernel (kept for comparison/debugging)."""


def _collision_distances_all(
    cfg: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
) -> Float[Array, "P"]:
    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    parts = [jnp.ravel(self_dists)]
    for wg in world_geoms:
        parts.append(jnp.ravel(robot_coll.compute_world_collision_distance(robot, cfg, wg)))
    return jnp.concatenate(parts)


def _collision_dists_reduced(
    cfg: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    temperature: float,
) -> Float[Array, "G"]:
    def smooth_min(d_flat: Array) -> Array:
        return -temperature * jax.scipy.special.logsumexp(-d_flat / temperature)

    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    groups = [smooth_min(jnp.ravel(self_dists))]
    for wg in world_geoms:
        dists = robot_coll.compute_world_collision_distance(robot, cfg, wg)
        groups.append(smooth_min(jnp.ravel(dists)))
    return jnp.stack(groups)


def _compute_coll_dists_and_jacs(
    trajs: Float[Array, "B T DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    temperature: float,
) -> tuple[Float[Array, "B T G"], Float[Array, "B T G DOF"]]:
    def per_cfg(cfg):
        d = _collision_dists_reduced(cfg, robot, robot_coll, world_geoms, temperature)
        J = jax.jacfwd(_collision_dists_reduced, argnums=0)(
            cfg, robot, robot_coll, world_geoms, temperature
        )
        return d, J

    return jax.vmap(jax.vmap(per_cfg))(trajs)


def _eval_cost(
    traj: Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    cfg: LsTrajOptConfig,
) -> Array:
    acc = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
    jerk = acc[1:] - acc[:-1]

    c = cfg.w_smooth * (
        cfg.w_acc * jnp.sum(acc**2)
        + cfg.w_jerk * jnp.sum(jerk**2)
    )

    viol_upper = jnp.maximum(0.0, traj - upper)
    viol_lower = jnp.maximum(0.0, lower - traj)
    c += cfg.w_limits * jnp.sum((viol_upper + viol_lower) ** 2)

    def per_step(q):
        d = _collision_distances_all(q, robot, robot_coll, world_geoms)
        return jnp.sum(-jnp.minimum(colldist_from_sdf(d, cfg.collision_margin), 0.0))

    c += cfg.w_collision_max * jnp.sum(jax.vmap(per_step)(traj))
    return c


def _solve_ls_subproblem_single(
    traj0: Float[Array, "T DOF"],
    q_ref: Float[Array, "T DOF"],
    d_k: Float[Array, "T G"],
    J_k: Float[Array, "T G DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    cfg: LsTrajOptConfig,
    w_coll: Array,
) -> Float[Array, "T DOF"]:
    T, dof = traj0.shape
    n = T * dof

    sqrt_w_acc = jnp.sqrt(cfg.w_smooth * cfg.w_acc)
    sqrt_w_jerk = jnp.sqrt(cfg.w_smooth * cfg.w_jerk)
    sqrt_w_trust = jnp.sqrt(cfg.w_trust)
    sqrt_w_limits = jnp.sqrt(cfg.w_limits)
    sqrt_w_coll = jnp.sqrt(w_coll)
    sqrt_w_endpoint = jnp.sqrt(cfg.w_endpoint)

    def residual_fn(x_flat: Float[Array, "n"]) -> Float[Array, "M"]:
        traj = x_flat.reshape(T, dof)

        acc = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
        jerk = acc[1:] - acc[:-1]

        delta = traj - q_ref
        d_lin = d_k + jnp.einsum("tgd,td->tg", J_k, delta)
        coll_viol = jnp.maximum(0.0, cfg.collision_margin - d_lin)

        lim_upper = jnp.maximum(0.0, traj - upper)
        lim_lower = jnp.maximum(0.0, lower - traj)

        parts = [
            (sqrt_w_acc * acc).reshape(-1),
            (sqrt_w_jerk * jerk).reshape(-1),
            (sqrt_w_trust * delta).reshape(-1),
            (sqrt_w_limits * lim_upper).reshape(-1),
            (sqrt_w_limits * lim_lower).reshape(-1),
            (sqrt_w_coll * coll_viol).reshape(-1),
            sqrt_w_endpoint * (traj[0] - start),
            sqrt_w_endpoint * (traj[-1] - goal),
        ]
        return jnp.concatenate(parts)

    def _cg_solve(matvec, b: Float[Array, "n"]) -> Float[Array, "n"]:
        x = jnp.zeros_like(b)
        r = b
        p = r
        rr = jnp.dot(r, r)
        bnorm = jnp.sqrt(jnp.dot(b, b) + 1e-18)
        tol2 = (cfg.cg_tol * bnorm + 1e-12) ** 2

        def body(_, carry):
            xk, rk, pk, rrk, active = carry

            def do_iter(_):
                Apk = matvec(pk)
                alpha = rrk / (jnp.dot(pk, Apk) + 1e-9)
                x_new = xk + alpha * pk
                r_new = rk - alpha * Apk
                rr_new = jnp.dot(r_new, r_new)
                beta = rr_new / (rrk + 1e-9)
                p_new = r_new + beta * pk
                still_active = rr_new > tol2
                return x_new, r_new, p_new, rr_new, still_active

            return jax.lax.cond(active, do_iter, lambda _: carry, operand=None)

        x, _, _, _, _ = jax.lax.fori_loop(
            0,
            cfg.n_cg_iters,
            body,
            (x, r, p, rr, jnp.bool_(True)),
        )
        return x

    x0 = traj0.reshape(-1)

    def lm_step(carry, _):
        x, lam, best_x, best_cost, stall, active = carry

        def do_step(_):
            r, jvp = jax.linearize(residual_fn, x)

            def jt(u):
                return jax.linear_transpose(jvp, x)(u)[0]

            curr_cost = jnp.dot(r, r)
            rhs = -jt(r)

            def matvec(v):
                return jt(jvp(v)) + lam * v

            delta = _cg_solve(matvec, rhs)

            delta = jnp.clip(delta, -cfg.max_delta_per_step, cfg.max_delta_per_step)

            def eval_alpha(alpha):
                x_new = x + alpha * delta
                r_new = residual_fn(x_new)
                return jnp.dot(r_new, r_new)

            alpha_costs = jax.vmap(eval_alpha)(_TRAJOPT_LS_ALPHAS)
            idx = jnp.argmin(alpha_costs)
            trial_cost = alpha_costs[idx]
            x_trial = x + _TRAJOPT_LS_ALPHAS[idx] * delta

            improved = trial_cost < curr_cost * (1.0 - 1e-4)
            x_out = jnp.where(improved, x_trial, x)
            lam_out = jnp.clip(
                jnp.where(improved, lam * 0.5, lam * 3.0),
                cfg.lambda_min,
                cfg.lambda_max,
            )

            best_x_out = jnp.where(trial_cost < best_cost, x_trial, best_x)
            best_cost_out = jnp.where(trial_cost < best_cost, trial_cost, best_cost)
            stall_out = jnp.where(improved, jnp.int32(0), stall + jnp.int32(1))
            active_out = stall_out < jnp.int32(cfg.inner_stall_patience)

            return (x_out, lam_out, best_x_out, best_cost_out, stall_out, active_out)

        return jax.lax.cond(active, do_step, lambda _: carry, operand=None), None

    x0 = x0.at[:dof].set(start)
    x0 = x0.at[(T - 1) * dof :].set(goal)

    r0 = residual_fn(x0)
    c0 = jnp.dot(r0, r0)

    (x_final, _, best_x, _, _, _), _ = jax.lax.scan(
        lm_step,
        (
            x0,
            jnp.array(cfg.lambda_init, dtype=x0.dtype),
            x0,
            c0,
            jnp.int32(0),
            jnp.bool_(True),
        ),
        None,
        length=cfg.n_ls_iters,
    )

    x_best = jnp.where(jnp.isfinite(jnp.sum(best_x)), best_x, x_final)
    traj = x_best.reshape(T, dof)
    traj = traj.at[0].set(start).at[-1].set(goal)
    return jnp.clip(traj, lower, upper).at[0].set(start).at[-1].set(goal)


@functools.partial(jax.jit, static_argnames=("opt_cfg",))
def _ls_trajopt_jax(
    init_trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: LsTrajOptConfig,
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    trajs = init_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)

    def outer_step(carry, _):
        trajs, w_coll = carry

        d_k, J_k = _compute_coll_dists_and_jacs(
            trajs,
            robot,
            robot_coll,
            world_geoms,
            opt_cfg.smooth_min_temperature,
        )

        new_trajs = jax.vmap(
            lambda traj, dk, Jk: _solve_ls_subproblem_single(
                traj,
                traj,
                dk,
                Jk,
                start,
                goal,
                lower,
                upper,
                opt_cfg,
                w_coll,
            )
        )(trajs, d_k, J_k)

        new_trajs = new_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)
        w_coll = jnp.minimum(w_coll * opt_cfg.penalty_scale, opt_cfg.w_collision_max)
        return (new_trajs, w_coll), None

    (final_trajs, _), _ = jax.lax.scan(
        outer_step,
        (trajs, jnp.array(opt_cfg.w_collision, dtype=jnp.float32)),
        None,
        length=opt_cfg.n_outer_iters,
    )

    costs = jax.vmap(
        lambda t: _eval_cost(t, lower, upper, robot, robot_coll, world_geoms, opt_cfg)
    )(final_trajs)
    best_idx = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]
    return best_traj, costs, final_trajs


def _link_pose_sequence(
    robot: Robot,
    traj: Float[Array, "T DOF"],
    link_idx: int,
) -> jaxlie.SE3:
    def per_cfg(cfg):
        Ts = robot.forward_kinematics(cfg)
        return jaxlie.SE3(Ts[link_idx]).wxyz_xyz

    return jaxlie.SE3(jax.vmap(per_cfg)(traj))


def _project_traj_with_jax_lsik(
    traj: Float[Array, "T DOF"],
    target_poses: jaxlie.SE3,
    robot: Robot,
    link_idx: int,
    cfg: LsTrajOptConfig,
    key: Array,
) -> Float[Array, "T DOF"]:
    qs = []
    prev = traj[0]
    keys = jax.random.split(key, traj.shape[0])
    for t in range(traj.shape[0]):
        prev = ls_ik_solve(
            robot=robot,
            target_link_indices=(link_idx,),
            target_poses=(jaxlie.SE3(target_poses.wxyz_xyz[t]),),
            rng_key=keys[t],
            previous_cfg=prev,
            num_seeds=cfg.ik_projection_num_seeds,
            max_iter=cfg.ik_projection_max_iter,
            continuity_weight=cfg.ik_projection_continuity_weight,
        )
        qs.append(prev)
    return jnp.stack(qs, axis=0)


def _project_batch_with_lsik(
    final_trajs: Float[Array, "B T DOF"],
    init_trajs: Float[Array, "B T DOF"],
    robot: Robot,
    opt_cfg: LsTrajOptConfig,
    use_cuda_projection: bool,
    key: Array,
) -> Float[Array, "B T DOF"]:
    link_idx = opt_cfg.ik_projection_link_index
    if link_idx is None:
        return final_trajs

    B = final_trajs.shape[0]
    keys = jax.random.split(key, B)
    out = []

    for b in range(B):
        target_poses = _link_pose_sequence(robot, init_trajs[b], link_idx)

        if use_cuda_projection:
            proj = ls_ik_solve_cuda_batch(
                robot=robot,
                target_link_indices=(link_idx,),
                target_poses=target_poses,
                rng_key=keys[b],
                previous_cfgs=final_trajs[b],
                num_seeds=opt_cfg.ik_projection_num_seeds,
                max_iter=opt_cfg.ik_projection_max_iter,
                continuity_weight=opt_cfg.ik_projection_continuity_weight,
                constraint_refine_iters=0,
            )
        else:
            proj = _project_traj_with_jax_lsik(
                final_trajs[b],
                target_poses,
                robot,
                link_idx,
                opt_cfg,
                keys[b],
            )

        proj = proj.at[0].set(final_trajs[b, 0]).at[-1].set(final_trajs[b, -1])
        out.append(proj)

    return jnp.stack(out, axis=0)


def ls_trajopt(
    init_trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: LsTrajOptConfig = LsTrajOptConfig(),
    *,
    use_cuda: bool = False,
    key: Array | None = None,
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """Least-squares trajectory optimization.

    The core solver stacks smoothness, collision, trust-region, limit, and
    endpoint residuals into one large least-squares objective over the full
    trajectory, then runs LM/Gauss-Newton updates with SCO-style collision
    re-linearization.

    To leverage existing LS-IK solvers as requested, set
    ik_projection_link_index in the config. The solver then runs an optional
    post-step trajectory projection stage:
    - JAX LS-IK projection when use_cuda is False.
    - CUDA LS-IK batch projection when use_cuda is True.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    if use_cuda and opt_cfg.use_legacy_cuda_kernel:
        from ..cuda_kernels._ls_trajopt_cuda import ls_trajopt_cuda

        best_traj, costs, final_trajs = ls_trajopt_cuda(
            init_trajs=init_trajs,
            start=start,
            goal=goal,
            robot=robot,
            robot_coll=robot_coll,
            world_geoms=world_geoms,
            opt_cfg=opt_cfg,
        )
    else:
        best_traj, costs, final_trajs = _ls_trajopt_jax(
            init_trajs=init_trajs,
            start=start,
            goal=goal,
            robot=robot,
            robot_coll=robot_coll,
            world_geoms=world_geoms,
            opt_cfg=opt_cfg,
        )

    final_trajs = _project_batch_with_lsik(
        final_trajs=final_trajs,
        init_trajs=init_trajs,
        robot=robot,
        opt_cfg=opt_cfg,
        use_cuda_projection=use_cuda,
        key=key,
    )

    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    costs = jax.vmap(
        lambda t: _eval_cost(t, lower, upper, robot, robot_coll, world_geoms, opt_cfg)
    )(final_trajs)
    best_idx = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]
    return best_traj, costs, final_trajs
