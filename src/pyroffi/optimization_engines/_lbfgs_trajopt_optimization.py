"""Direct L-BFGS trajectory optimization.

Unlike SCO which linearizes collision constraints and solves a convex
subproblem, this solver runs L-BFGS directly on the full nonlinear cost:

    min  w_smooth * J_smooth(q) + w_collision * Σ coll_cost(q_t) + w_limits * J_limits(q)

This mirrors cuRobo's L-BFGS refinement strategy: gradient descent on
the true (non-convex) objective with penalty continuation on the collision
weight.

Start/goal endpoints are pinned via gradient masking.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from .._robot import Robot
from ..collision import RobotCollision, colldist_from_sdf
from ._sco_optimization import (
    _lbfgs_two_loop,
    _smoothness_cost,
    _limits_cost,
    _collision_distances_all,
    _LS_ALPHAS,
)


@dataclass(frozen=True)
class LbfgsTrajOptConfig:
    """Hyper-parameters for the direct L-BFGS trajectory optimizer."""

    n_iters: int = 100
    """Total number of L-BFGS iterations."""

    m_lbfgs: int = 6
    """L-BFGS history size (number of curvature pairs)."""

    # --- Smoothness ---
    w_smooth: float = 1.0
    w_vel: float = 1.0
    w_acc: float = 0.5
    w_jerk: float = 0.1

    # --- Collision ---
    w_collision: float = 1.0
    """Initial collision penalty weight."""

    w_collision_max: float = 100.0
    """Maximum collision penalty weight after continuation."""

    penalty_scale: float = 2.0
    """Multiplicative increase in w_collision per escalation step."""

    escalation_interval: int = 20
    """Increase w_collision every this many iterations."""

    collision_margin: float = 0.01
    """Activation margin for collision cost (metres)."""

    # --- Joint limits ---
    w_limits: float = 1.0


# ---------------------------------------------------------------------------
# Single-trajectory L-BFGS solver
# ---------------------------------------------------------------------------

def _lbfgs_direct_solve(
    traj:        Float[Array, "T DOF"],
    lower:       Float[Array, "DOF"],
    upper:       Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    cfg:         LbfgsTrajOptConfig,
) -> Float[Array, "T DOF"]:
    """Run L-BFGS directly on the nonlinear trajectory cost."""
    m   = cfg.m_lbfgs
    T   = traj.shape[0]
    DOF = traj.shape[1]
    n   = T * DOF

    # Mask that zeros gradients at first/last waypoints (endpoint pinning)
    endpoint_mask = (
        jnp.ones(n)
        .at[:DOF].set(0.0)
        .at[n - DOF:].set(0.0)
    )

    def cost_fn(x_flat: Float[Array, "n"], w_coll: Array) -> Array:
        t = x_flat.reshape(T, DOF)

        # Smoothness
        c = cfg.w_smooth * _smoothness_cost(t, cfg.w_vel, cfg.w_acc, cfg.w_jerk)

        # Joint limits
        c += cfg.w_limits * _limits_cost(t, lower, upper)

        # True nonlinear collision cost (not linearized)
        def per_step_coll(q):
            dists = _collision_distances_all(q, robot, robot_coll, world_geoms)
            return jnp.sum(-jnp.minimum(colldist_from_sdf(dists, cfg.collision_margin), 0.0))

        c += w_coll * jnp.sum(jax.vmap(per_step_coll)(t))

        return c

    x0 = traj.reshape(n)
    w_coll0 = jnp.array(cfg.w_collision, dtype=jnp.float32)
    cost0, g0 = jax.value_and_grad(cost_fn)(x0, w_coll0)
    g0 = g0 * endpoint_mask

    init_carry = (
        x0,                       # current iterate
        x0,                       # best iterate
        cost0,                    # best cost
        x0,                       # x_prev
        g0,                       # g_prev
        jnp.zeros((m, n)),        # s_buf
        jnp.zeros((m, n)),        # y_buf
        jnp.zeros(m),             # rho_buf
        jnp.int32(0),             # m_used
        jnp.int32(0),             # newest
        jnp.int32(0),             # iter_count
        w_coll0,                  # current collision weight
    )

    def lbfgs_step(carry, _):
        (x, best_x, best_cost,
         x_prev, g_prev,
         s_buf, y_buf, rho_buf,
         m_used, newest, iter_count,
         w_coll) = carry

        # Escalate collision weight on schedule
        w_coll = jnp.where(
            (iter_count > 0) & (iter_count % cfg.escalation_interval == 0),
            jnp.minimum(w_coll * cfg.penalty_scale, cfg.w_collision_max),
            w_coll,
        )

        cost_val, g = jax.value_and_grad(cost_fn)(x, w_coll)
        g = g * endpoint_mask

        # Update L-BFGS curvature history
        s_k = x - x_prev
        y_k = g - g_prev
        sy  = jnp.dot(s_k, y_k)
        yy  = jnp.dot(y_k, y_k)
        valid = (sy > 1e-10 * yy + 1e-30) & (iter_count > 0)

        new_newest   = (newest + 1) % m
        actual_newest = jnp.where(valid, new_newest, newest)
        s_buf   = jnp.where(valid, s_buf.at[new_newest].set(s_k), s_buf)
        y_buf   = jnp.where(valid, y_buf.at[new_newest].set(y_k), y_buf)
        rho_buf = jnp.where(valid, rho_buf.at[new_newest].set(1.0 / (sy + 1e-30)), rho_buf)
        m_used  = jnp.where(valid & (m_used < m), m_used + 1, m_used)
        newest  = actual_newest

        # L-BFGS search direction
        dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
        dir_gd    = -g / (jnp.linalg.norm(g) + 1e-18)
        direction = jnp.where(m_used > 0, dir_lbfgs, dir_gd)
        direction = direction * endpoint_mask

        # Line search
        suff_thresh = cost_val * (1.0 - 1e-4)
        trial_costs = jax.vmap(lambda a: cost_fn(x + a * direction, w_coll))(_LS_ALPHAS)
        has_suff    = trial_costs < suff_thresh
        best_idx    = jnp.where(
            jnp.any(has_suff),
            jnp.argmax(has_suff),
            jnp.argmin(trial_costs),
        )
        alpha    = _LS_ALPHAS[best_idx]
        x_new    = x + alpha * direction
        new_cost = trial_costs[best_idx]

        # Evaluate cost at max collision weight for ranking
        eval_cost = cost_fn(x_new, jnp.array(cfg.w_collision_max, dtype=jnp.float32))
        improved  = eval_cost < best_cost
        best_x    = jnp.where(improved, x_new, best_x)
        best_cost = jnp.where(improved, eval_cost, best_cost)

        return (
            x_new, best_x, best_cost,
            x, g,
            s_buf, y_buf, rho_buf,
            m_used, newest, iter_count + 1,
            w_coll,
        ), None

    (_, best_x, _, _, _, _, _, _, _, _, _, _), _ = jax.lax.scan(
        lbfgs_step, init_carry, None, length=cfg.n_iters,
    )

    return best_x.reshape(T, DOF)


# ---------------------------------------------------------------------------
# Final nonlinear cost (for ranking)
# ---------------------------------------------------------------------------

def _eval_cost_lbfgs(
    traj:        Float[Array, "T DOF"],
    lower:       Float[Array, "DOF"],
    upper:       Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    cfg:         LbfgsTrajOptConfig,
) -> Array:
    """Full nonlinear cost at maximum collision weight."""
    cost = cfg.w_smooth * _smoothness_cost(traj, cfg.w_vel, cfg.w_acc, cfg.w_jerk)
    cost += cfg.w_limits * _limits_cost(traj, lower, upper)

    def per_step(c):
        dists = _collision_distances_all(c, robot, robot_coll, world_geoms)
        return jnp.sum(-jnp.minimum(colldist_from_sdf(dists, cfg.collision_margin), 0.0))

    cost += cfg.w_collision_max * jnp.sum(jax.vmap(per_step)(traj))
    return cost


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("opt_cfg",),
)
def _lbfgs_trajopt_jax(
    init_trajs:  Float[Array, "B T DOF"],
    start:       Float[Array, "DOF"],
    goal:        Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    opt_cfg:     LbfgsTrajOptConfig = LbfgsTrajOptConfig(),
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    # Pin endpoints
    trajs = init_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)

    # Solve all trajectories in parallel
    final_trajs = jax.vmap(
        lambda traj: _lbfgs_direct_solve(
            traj, lower, upper, robot, robot_coll, world_geoms, opt_cfg
        )
    )(trajs)

    # Re-pin endpoints
    final_trajs = final_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)

    # Rank by full nonlinear cost
    costs = jax.vmap(
        lambda t: _eval_cost_lbfgs(t, lower, upper, robot, robot_coll, world_geoms, opt_cfg)
    )(final_trajs)
    best_idx  = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]

    return best_traj, costs, final_trajs


def lbfgs_trajopt(
    init_trajs:  Float[Array, "B T DOF"],
    start:       Float[Array, "DOF"],
    goal:        Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    opt_cfg:     LbfgsTrajOptConfig = LbfgsTrajOptConfig(),
    *,
    use_cuda:    bool = False,
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """Direct L-BFGS trajectory optimization.

    Runs L-BFGS directly on the nonlinear cost (smoothness + true collision
    distances + joint limits), with penalty continuation on the collision
    weight.  This mirrors cuRobo's L-BFGS refinement approach.

    Args:
        init_trajs:  Initial trajectory batch.  Shape [B, T, DOF].
        start:       Start joint configuration.  Shape [DOF].
        goal:        Goal joint configuration.   Shape [DOF].
        robot:       Robot kinematics pytree.
        robot_coll:  Robot collision model pytree.
        world_geoms: Tuple of stacked world collision geometry objects.
        opt_cfg:     Hyper-parameters (static — changes trigger recompilation).
        use_cuda:    Reserved for future CUDA backend.

    Returns:
        best_traj:   Trajectory with lowest final nonlinear cost. [T, DOF].
        costs:       Final nonlinear cost per trajectory.         [B].
        final_trajs: All optimized trajectories.                  [B, T, DOF].
    """
    return _lbfgs_trajopt_jax(
        init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg
    )
