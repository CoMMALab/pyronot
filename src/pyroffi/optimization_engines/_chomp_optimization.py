"""CHOMP TrajOpt: Covariant Hamiltonian Optimization for Motion Planning.

This module provides a CHOMP-style trajectory optimizer with a public API
matching ``sco_trajopt`` so callers can switch solvers with minimal changes.

Update rule (single trajectory, endpoints pinned):

    q_{k+1} = q_k - alpha * M^{-1} grad J(q_k)

where ``M`` is a smoothness metric built from a second-difference precision
matrix over time (covariant gradient preconditioning).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ..collision import RobotCollision


_LS_ALPHAS = jnp.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.0])


@dataclass(frozen=True)
class ChompTrajOptConfig:
    """Hyper-parameters for CHOMP trajectory optimization."""

    n_iters: int = 100
    """Number of CHOMP update iterations."""

    step_size: float = 0.05
    """Base step size used by the line search candidates."""

    w_smooth: float = 1.0
    """Overall smoothness weight."""

    w_vel: float = 1.0
    """Unused; kept for API compatibility with SCO config."""

    w_acc: float = 0.5
    """Relative weight of acceleration within smoothness."""

    w_jerk: float = 0.1
    """Relative weight of jerk within smoothness."""

    w_collision: float = 3.0
    """Initial collision penalty weight."""

    w_collision_max: float = 50.0
    """Maximum collision penalty weight for continuation."""

    collision_penalty_scale: float = 1.05
    """Per-iteration multiplicative scaling for collision penalty."""

    collision_margin: float = 0.01
    """Activation margin for collision hinge loss (metres)."""

    w_limits: float = 1.0
    """Weight for soft joint-limit penalty."""

    use_covariant_update: bool = True
    """If True, precondition gradient by inverse smoothness metric."""

    smoothness_reg: float = 1e-3
    """Diagonal regularizer added before inverting smoothness metric."""

    n_cg_iters: int = 24
    """Conjugate-gradient iterations for matrix-free covariant preconditioning."""

    cg_tol: float = 1e-4
    """Relative tolerance used by the matrix-free covariant CG solve."""

    grad_clip_norm: float = 10.0
    """Global gradient norm clip (0 disables clipping)."""

    max_delta_per_step: float = 0.05
    """Trust-region style clamp on per-joint update per iteration (radians)."""

    early_stop_patience: int = 15
    """Stop iterating when no meaningful improvement is seen for this many steps."""

    min_cost_improve: float = 1e-5
    """Minimum decrease in cost to count as improvement."""


def _smoothness_cost(
    traj: Float[Array, "T DOF"],
    w_vel: float,
    w_acc: float,
    w_jerk: float,
) -> Array:
    """4th-order central-difference acceleration + jerk smoothness cost."""
    acc = (
        -      traj[:-4]
        + 16.0 * traj[1:-3]
        - 30.0 * traj[2:-2]
        + 16.0 * traj[3:-1]
        -      traj[4:]
    ) / 12.0
    jerk = acc[1:] - acc[:-1]
    cost = w_acc * jnp.sum(acc ** 2)
    cost += w_jerk * jnp.sum(jerk ** 2)
    return cost


def _limits_cost(
    traj: Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
) -> Array:
    """Squared exceedance penalty for joint-limit violations."""
    viol_upper = jnp.maximum(0.0, traj - upper)
    viol_lower = jnp.maximum(0.0, lower - traj)
    return jnp.sum((viol_upper + viol_lower) ** 2)


def _collision_distances_all(
    cfg: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
) -> Float[Array, "P"]:
    """Flat concatenation of all self + world collision distances."""
    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    parts = [jnp.ravel(self_dists)]
    for wg in world_geoms:
        parts.append(jnp.ravel(
            robot_coll.compute_world_collision_distance(robot, cfg, wg)
        ))
    return jnp.concatenate(parts)


def _collision_cost(
    traj: Float[Array, "T DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    margin: float,
) -> Array:
    """Quadratic hinge penalty on signed distances along trajectory."""
    def per_step(c):
        dists = _collision_distances_all(c, robot, robot_coll, world_geoms)
        viol = jnp.maximum(0.0, margin - dists)
        return jnp.sum(viol ** 2)

    return jnp.sum(jax.vmap(per_step)(traj))


def _chomp_metric_apply_1d(
    x: Float[Array, "Tin"],
    reg: float,
) -> Float[Array, "Tin"]:
    """Apply the interior CHOMP smoothness precision operator without materializing it."""
    y = 6.0 * x
    y = y.at[1:].add(-4.0 * x[:-1])
    y = y.at[:-1].add(-4.0 * x[1:])
    y = y.at[2:].add(x[:-2])
    y = y.at[:-2].add(x[2:])
    return y + reg * x


def _cg_solve_spd(
    matvec,
    b: Float[Array, "n"],
    n_iters: int,
    tol: float,
) -> Float[Array, "n"]:
    """Truncated CG solve for symmetric positive-definite linear systems."""
    x = jnp.zeros_like(b)
    r = b
    p = r
    rr = jnp.dot(r, r)
    bnorm = jnp.sqrt(jnp.dot(b, b) + 1e-18)
    tol2 = (tol * bnorm + 1e-12) ** 2

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
        n_iters,
        body,
        (x, r, p, rr, jnp.bool_(True)),
    )
    return x


def _precondition_direction(
    grad: Float[Array, "T DOF"],
    smoothness_reg: float,
    n_cg_iters: int,
    cg_tol: float,
) -> Float[Array, "T DOF"]:
    """Apply CHOMP covariant metric inverse via matrix-free CG on each DOF timeline."""
    if grad.shape[0] <= 2:
        return -grad

    interior_grad = grad[1:-1]  # [Tin, DOF]
    solve_one_dof = lambda g: _cg_solve_spd(
        lambda v: _chomp_metric_apply_1d(v, smoothness_reg),
        g,
        n_cg_iters,
        cg_tol,
    )
    interior_dir = -jax.vmap(solve_one_dof, in_axes=1, out_axes=1)(interior_grad)

    direction = jnp.zeros_like(grad)
    direction = direction.at[1:-1].set(interior_dir)
    return direction


def _optimize_single_traj(
    init_traj: Float[Array, "T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: ChompTrajOptConfig,
) -> Float[Array, "T DOF"]:
    """Run CHOMP iterations on one trajectory."""
    traj = init_traj.at[0].set(start).at[-1].set(goal)

    def cost_fn(t: Float[Array, "T DOF"], w_coll: Array) -> Array:
        c = opt_cfg.w_smooth * _smoothness_cost(t, opt_cfg.w_vel, opt_cfg.w_acc, opt_cfg.w_jerk)
        c += opt_cfg.w_limits * _limits_cost(t, lower, upper)
        c += w_coll * _collision_cost(
            t, robot, robot_coll, world_geoms, opt_cfg.collision_margin
        )
        return c

    w_coll0 = jnp.array(opt_cfg.w_collision, dtype=jnp.float32)
    init_cost = cost_fn(traj, w_coll0)

    def step_fn(carry, _):
        curr_traj, w_coll, best_cost, stall_steps, active = carry

        def do_step(state):
            curr_traj, w_coll, best_cost, stall_steps = state

            curr_cost, grad = jax.value_and_grad(cost_fn, argnums=0)(curr_traj, w_coll)
            grad_norm = jnp.linalg.norm(grad)
            clip_scale = jnp.where(
                (opt_cfg.grad_clip_norm > 0.0) & (grad_norm > opt_cfg.grad_clip_norm),
                opt_cfg.grad_clip_norm / (grad_norm + 1e-12),
                1.0,
            )
            grad = (grad * clip_scale).at[0].set(0.0).at[-1].set(0.0)

            if opt_cfg.use_covariant_update:
                direction = _precondition_direction(
                    grad,
                    opt_cfg.smoothness_reg,
                    opt_cfg.n_cg_iters,
                    opt_cfg.cg_tol,
                )
            else:
                direction = -grad

            alphas = opt_cfg.step_size * _LS_ALPHAS

            def apply_alpha(a):
                delta = jnp.clip(
                    a * direction,
                    -opt_cfg.max_delta_per_step,
                    opt_cfg.max_delta_per_step,
                )
                t_new = curr_traj + delta
                return t_new.at[0, :].set(start).at[-1, :].set(goal)

            trial_trajs = jax.vmap(apply_alpha)(alphas)
            trial_costs = jax.vmap(lambda t: cost_fn(t, w_coll))(trial_trajs)

            best_idx = jnp.argmin(trial_costs)
            best_trial = trial_trajs[best_idx]
            best_trial_cost = trial_costs[best_idx]
            improved = best_trial_cost < (curr_cost - opt_cfg.min_cost_improve)
            next_traj = jnp.where(improved, best_trial, curr_traj)

            best_cost = jnp.minimum(best_cost, best_trial_cost)
            stall_steps = jnp.where(improved, jnp.int32(0), stall_steps + 1)
            active_next = stall_steps < opt_cfg.early_stop_patience

            w_coll = jnp.minimum(
                w_coll * opt_cfg.collision_penalty_scale,
                jnp.array(opt_cfg.w_collision_max, dtype=jnp.float32),
            )
            return next_traj, w_coll, best_cost, stall_steps, active_next

        def no_step(state):
            curr_traj, w_coll, best_cost, stall_steps = state
            return curr_traj, w_coll, best_cost, stall_steps, jnp.bool_(False)

        next_traj, w_coll, best_cost, stall_steps, active = jax.lax.cond(
            active,
            do_step,
            no_step,
            (curr_traj, w_coll, best_cost, stall_steps),
        )
        return (next_traj, w_coll, best_cost, stall_steps, active), None

    (final_traj, _, _, _, _), _ = jax.lax.scan(
        step_fn,
        (traj, w_coll0, init_cost, jnp.int32(0), jnp.bool_(True)),
        None,
        length=opt_cfg.n_iters,
    )
    return final_traj


def _eval_cost(
    traj: Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: ChompTrajOptConfig,
) -> Array:
    """Final nonlinear cost used for ranking trajectories."""
    c = opt_cfg.w_smooth * _smoothness_cost(traj, opt_cfg.w_vel, opt_cfg.w_acc, opt_cfg.w_jerk)
    c += opt_cfg.w_limits * _limits_cost(traj, lower, upper)
    c += opt_cfg.w_collision_max * _collision_cost(
        traj, robot, robot_coll, world_geoms, opt_cfg.collision_margin
    )
    return c


@functools.partial(jax.jit, static_argnames=("opt_cfg",))
def _chomp_trajopt_jax(
    init_trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: ChompTrajOptConfig = ChompTrajOptConfig(),
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """Batch CHOMP optimizer entrypoint."""
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    final_trajs = jax.vmap(
        lambda t: _optimize_single_traj(
            t, start, goal, lower, upper, robot, robot_coll, world_geoms, opt_cfg
        )
    )(init_trajs)

    final_trajs = final_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)

    costs = jax.vmap(
        lambda t: _eval_cost(t, lower, upper, robot, robot_coll, world_geoms, opt_cfg)
    )(final_trajs)
    best_idx = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]
    return best_traj, costs, final_trajs


def chomp_trajopt(
    init_trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: ChompTrajOptConfig = ChompTrajOptConfig(),
    *,
    use_cuda: bool = False,
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """CHOMP trajectory optimization.

    Args:
        init_trajs:  Initial trajectory batch.  Shape [B, T, DOF].
        start:       Start joint configuration.  Shape [DOF].
        goal:        Goal joint configuration.   Shape [DOF].
        robot:       Robot kinematics pytree.
        robot_coll:  Robot collision model pytree.
        world_geoms: Tuple of stacked world collision geometry objects.
        opt_cfg:     CHOMP hyper-parameters (static under JIT).
        use_cuda:    Reserved for API compatibility; not implemented yet.

    Returns:
        best_traj:   Trajectory with lowest final cost. [T, DOF].
        costs:       Final cost per trajectory.         [B].
        final_trajs: All optimized trajectories.        [B, T, DOF].
    """
    if use_cuda:
        from ..cuda_kernels._chomp_trajopt_cuda import chomp_trajopt_cuda
        return chomp_trajopt_cuda(
            init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg
        )
    return _chomp_trajopt_jax(
        init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg
    )
