"""SCO-style Batched Trajectory Optimization.

JAX-native implementation of a batched trajectory optimization (TrajOpt) pipeline
loosely based on curobo's accelerated TrajOpt.  All B candidate trajectories are
optimized in parallel using ``jax.vmap`` and ``jax.lax.scan``.

Pipeline
--------
1. Initialize trajectories  [B, T, DOF]
2. Run batched gradient descent  (lax.scan over N_iters steps)
3. Evaluate costs  [B]
4. Return best trajectory + all costs

Cost structure
--------------
  J = w_smooth     * J_smooth        (velocity + acceleration + jerk)
    + w_collision  * J_collision     (self + world, via colldist_from_sdf)
    + w_goal       * J_goal          (final-state distance to goal)
    + w_limits     * J_limits        (soft joint-limit penalty)

Optimization
------------
Gradient descent with optional parallel line-search over ``_LS_ALPHAS``.
The entire pipeline is wrapped in ``jax.jit`` and operates on static shapes,
making it fully JIT-compilable and XLA-optimizable.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ..collision import CollGeom, RobotCollision, colldist_from_sdf

# ---------------------------------------------------------------------------
# Step-size candidates (same convention as IK solvers)
# ---------------------------------------------------------------------------

_LS_ALPHAS: Float[Array, "5"] = jnp.array([1.0, 0.5, 0.25, 0.1, 0.025])


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajOptConfig:
    """Hyper-parameters for the SCO TrajOpt solver."""

    # --- Iteration budget ---
    n_iters: int = 100
    """Number of gradient-descent iterations."""

    # --- Learning rate ---
    lr: float = 0.01
    """Base learning rate (gradient step size)."""

    use_line_search: bool = True
    """If True, run a parallel line-search at each step and keep the best alpha."""

    # --- Cost weights ---
    w_smooth: float = 1.0
    """Weight for the smoothness cost (vel + acc + jerk)."""

    w_vel: float = 1.0
    """Relative weight of velocity within smoothness."""

    w_acc: float = 0.5
    """Relative weight of acceleration within smoothness."""

    w_jerk: float = 0.1
    """Relative weight of jerk within smoothness."""

    w_collision: float = 10.0
    """Weight for the collision cost."""

    collision_margin: float = 0.01
    """Activation distance (margin) for colldist_from_sdf."""

    w_goal: float = 5.0
    """Weight for the goal cost (distance of last waypoint to target)."""

    w_limits: float = 1.0
    """Weight for joint-limit violation penalty."""


# ---------------------------------------------------------------------------
# Individual cost components  (operate on a single trajectory [T, DOF])
# ---------------------------------------------------------------------------

def _smoothness_cost(
    traj: Float[Array, "T DOF"],
    w_vel: float,
    w_acc: float,
    w_jerk: float,
) -> Array:
    """Sum of squared finite-difference norms (vel, acc, jerk)."""
    vel  = traj[1:] - traj[:-1]                 # [T-1, DOF]
    acc  = vel[1:] - vel[:-1]                   # [T-2, DOF]
    jerk = acc[1:] - acc[:-1]                   # [T-3, DOF]

    cost  = w_vel  * jnp.sum(vel  ** 2)
    cost += w_acc  * jnp.sum(acc  ** 2)
    cost += w_jerk * jnp.sum(jerk ** 2)
    return cost


def _goal_cost(
    traj: Float[Array, "T DOF"],
    goal: Float[Array, "DOF"],
) -> Array:
    """Squared L2 distance between the final waypoint and the goal configuration."""
    return jnp.sum((traj[-1] - goal) ** 2)


def _limits_cost(
    traj: Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
) -> Array:
    """Soft penalty for joint-limit violations (sum of squared exceedances)."""
    viol_upper = jnp.maximum(0.0, traj - upper)   # [T, DOF]
    viol_lower = jnp.maximum(0.0, lower - traj)   # [T, DOF]
    return jnp.sum((viol_upper + viol_lower) ** 2)


def _collision_cost_single_cfg(
    cfg: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geom: CollGeom | None,
    margin: float,
) -> Array:
    """Collision cost for a single configuration (self + world)."""
    cost = jnp.zeros(())

    # Self-collision
    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    cost += jnp.sum(-jnp.minimum(colldist_from_sdf(self_dists, margin), 0.0))

    # World collision
    if world_geom is not None:
        world_dists = robot_coll.compute_world_collision_distance(
            robot, cfg, world_geom
        )
        cost += jnp.sum(-jnp.minimum(colldist_from_sdf(world_dists, margin), 0.0))

    return cost


def _collision_cost_traj(
    traj: Float[Array, "T DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geom: CollGeom | None,
    margin: float,
) -> Array:
    """Sum of collision costs over all T timesteps."""
    per_step = jax.vmap(
        _collision_cost_single_cfg,
        in_axes=(0, None, None, None, None),
    )(traj, robot, robot_coll, world_geom, margin)
    return jnp.sum(per_step)


# ---------------------------------------------------------------------------
# Combined cost  (single trajectory)
# ---------------------------------------------------------------------------

def _total_cost(
    traj: Float[Array, "T DOF"],
    goal: Float[Array, "DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geom: CollGeom | None,
    cfg: TrajOptConfig,
) -> Array:
    """Scalar cost for a single trajectory."""
    cost = jnp.zeros(())

    cost += cfg.w_smooth * _smoothness_cost(
        traj, cfg.w_vel, cfg.w_acc, cfg.w_jerk
    )
    cost += cfg.w_goal * _goal_cost(traj, goal)
    cost += cfg.w_limits * _limits_cost(traj, lower, upper)
    cost += cfg.w_collision * _collision_cost_traj(
        traj, robot, robot_coll, world_geom, cfg.collision_margin
    )

    return cost


# ---------------------------------------------------------------------------
# Batched cost + grad  (over B trajectories)
# ---------------------------------------------------------------------------

def _make_batched_fns(
    goal: Float[Array, "DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geom: CollGeom | None,
    opt_cfg: TrajOptConfig,
):
    """Return (batched_cost_fn, batched_grad_fn) closed over all static data."""

    def single_cost(traj):
        return _total_cost(
            traj, goal, lower, upper, robot, robot_coll, world_geom, opt_cfg
        )

    batched_cost_fn = jax.vmap(single_cost, in_axes=0)    # [B, T, DOF] -> [B]
    batched_grad_fn = jax.vmap(jax.grad(single_cost), in_axes=0)  # [B, T, DOF] -> [B, T, DOF]
    return batched_cost_fn, batched_grad_fn


# ---------------------------------------------------------------------------
# Optimization step
# ---------------------------------------------------------------------------

def _make_step_fn(
    batched_cost_fn,
    batched_grad_fn,
    lr: float,
    use_line_search: bool,
):
    """Return a single lax.scan-compatible step function."""

    def step(trajs: Float[Array, "B T DOF"], _) -> tuple[Float[Array, "B T DOF"], None]:
        grads = batched_grad_fn(trajs)          # [B, T, DOF]

        if use_line_search:
            # Evaluate each alpha in parallel: alphas shape [A]
            # candidates shape [A, B, T, DOF]
            candidates = (
                trajs[None, ...]
                - _LS_ALPHAS[:, None, None, None] * lr * grads[None, ...]
            )
            # costs shape [A, B]
            costs = jax.vmap(batched_cost_fn)(candidates)
            # best alpha index per trajectory: [B]
            best_alpha_idx = jnp.argmin(costs, axis=0)
            # Gather: for each trajectory pick the best candidate
            new_trajs = candidates[best_alpha_idx, jnp.arange(trajs.shape[0])]
        else:
            new_trajs = trajs - lr * grads

        return new_trajs, None

    return step


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("opt_cfg",),
)
def sco_trajopt(
    init_trajs: Float[Array, "B T DOF"],
    goal: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geom: CollGeom | None,
    opt_cfg: TrajOptConfig = TrajOptConfig(),
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """Batched SCO-style trajectory optimization.

    Optimizes all B candidate trajectories in parallel and returns the best one.

    Args:
        init_trajs:  Initial trajectory batch.  Shape ``[B, T, DOF]``.
        goal:        Target joint configuration for the final waypoint.  Shape ``[DOF]``.
        robot:       Robot kinematics pytree.
        robot_coll:  Robot collision model pytree.
        world_geom:  World collision geometry (or ``None`` to skip world collisions).
        opt_cfg:     Hyper-parameters (static — changing them triggers recompilation).

    Returns:
        best_traj:   Optimized trajectory with lowest final cost.  Shape ``[T, DOF]``.
        costs:       Final cost per trajectory in the batch.  Shape ``[B]``.
        final_trajs: All optimized trajectories.  Shape ``[B, T, DOF]``.
    """
    lower = robot.joints.lower_limits   # [DOF]
    upper = robot.joints.upper_limits   # [DOF]

    batched_cost_fn, batched_grad_fn = _make_batched_fns(
        goal, lower, upper, robot, robot_coll, world_geom, opt_cfg
    )

    step_fn = _make_step_fn(
        batched_cost_fn, batched_grad_fn, opt_cfg.lr, opt_cfg.use_line_search
    )

    final_trajs, _ = jax.lax.scan(step_fn, init_trajs, None, length=opt_cfg.n_iters)

    costs = batched_cost_fn(final_trajs)       # [B]
    best_idx = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]          # [T, DOF]

    return best_traj, costs, final_trajs


# ---------------------------------------------------------------------------
# Convenience: initialise trajectory batch by interpolation
# ---------------------------------------------------------------------------

def make_init_trajs(
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    n_batch: int,
    n_timesteps: int,
    key: Array,
    noise_scale: float = 0.05,
) -> Float[Array, "B T DOF"]:
    """Create a batch of linearly-interpolated trajectories with small random noise.

    Args:
        start:        Start joint configuration.
        goal:         Goal joint configuration.
        n_batch:      Number of candidate trajectories.
        n_timesteps:  Number of waypoints (including start and end).
        key:          JAX PRNG key.
        noise_scale:  Standard deviation of additive Gaussian noise.

    Returns:
        Trajectory batch of shape ``[B, T, DOF]``.
    """
    # Linear interpolation: [T, DOF]
    t = jnp.linspace(0.0, 1.0, n_timesteps)[:, None]   # [T, 1]
    base = start[None, :] * (1.0 - t) + goal[None, :] * t  # [T, DOF]

    # Tile across batch and add noise
    trajs = jnp.broadcast_to(base[None], (n_batch, n_timesteps, start.shape[0]))
    noise = jax.random.normal(key, trajs.shape) * noise_scale
    return trajs + noise
