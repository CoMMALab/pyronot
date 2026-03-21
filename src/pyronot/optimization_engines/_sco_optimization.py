"""SCO TrajOpt: Sequential Convex Optimization for trajectory planning.

SCO pipeline (Schulman et al. 2013, "Finding Locally Optimal,
Collision-Free Trajectories with Sequential Convex Optimization"):

  Outer loop (n_outer_iters):
    1. Linearize collision constraints at the current trajectory q_k:
           d_lin(q_t) = d(q_k_t) + J_d(q_k_t) @ (q_t - q_k_t)
       Jacobians are computed once per outer iteration via jax.jacobian.
    2. Solve the inner convex subproblem with L-BFGS (n_inner_iters steps):
           min  w_smooth  * J_smooth(q)               [exact quadratic]
              + w_coll    * Σ max(0, margin-d_lin(q))² [convex hinge]
              + w_trust   * ||q - q_k||²               [trust region]
              + w_limits  * J_limits(q)                [quadratic]
       Start/goal endpoints are pinned via gradient masking.
    3. Set q_k ← solution.
    4. Scale w_coll by penalty_scale (penalty continuation).

The key distinction from plain gradient descent is that the non-convex
collision distances are *linearized* at each outer iterate and the
resulting convex subproblem is solved to near-optimality with L-BFGS,
rather than taking a single gradient step on the nonlinear objective.
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

# ---------------------------------------------------------------------------
# Line-search step sizes (mirrors _ik_primitives._LS_ALPHAS)
# ---------------------------------------------------------------------------

_LS_ALPHAS = jnp.array([1.0, 0.5, 0.25, 0.1, 0.025])


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajOptConfig:
    """Hyper-parameters for the SCO TrajOpt solver."""

    # --- Outer SCO loop ---
    n_outer_iters:   int   = 10
    """Number of linearize-and-solve outer iterations."""

    # --- Inner L-BFGS solver ---
    n_inner_iters:   int   = 30
    """L-BFGS steps per outer iteration."""

    m_lbfgs:         int   = 6
    """L-BFGS history size (number of curvature pairs)."""

    # --- Smoothness cost weights ---
    w_smooth:        float = 1.0
    """Overall smoothness weight."""

    w_vel:           float = 1.0
    """Unused; kept for API compatibility."""

    w_acc:           float = 0.5
    """Relative weight of acceleration within smoothness."""

    w_jerk:          float = 0.1
    """Relative weight of jerk within smoothness."""

    # --- Collision ---
    w_collision:     float = 1.0
    """Initial collision penalty weight (increased each outer iteration)."""

    w_collision_max: float = 100.0
    """Maximum collision penalty weight after continuation."""

    penalty_scale:   float = 3.0
    """Multiplicative increase in w_collision per outer iteration."""

    collision_margin: float = 0.01
    """Activation margin for the collision cost (metres)."""

    # --- Trust region ---
    w_trust:         float = 0.5
    """Penalty weight for deviating from the linearization point."""

    # --- Joint limits ---
    w_limits:        float = 1.0
    """Weight for the soft joint-limit violation penalty."""

    # --- Collision dimensionality reduction ---
    smooth_min_temperature: float = 0.05
    """Temperature for the per-group smooth-minimum aggregation.

    Instead of keeping all P raw distances (potentially hundreds), one
    smooth-minimum scalar is computed per collision group (self-collision +
    one per world-geometry type).  This reduces the Jacobian from [P, DOF]
    to [n_groups, DOF], cutting Jacobian memory and compile time by ~50-100x.

    Smaller temperature → closer to the true minimum but steeper gradients.
    """


# ---------------------------------------------------------------------------
# L-BFGS two-loop recursion  (Nocedal; self-contained copy)
# ---------------------------------------------------------------------------

def _lbfgs_two_loop(
    g:       Float[Array, "n"],
    s_buf:   Float[Array, "m n"],
    y_buf:   Float[Array, "m n"],
    rho_buf: Float[Array, "m"],
    m_used:  Array,          # traced int32
    newest:  Array,          # traced int32
    m_lbfgs: int,            # static Python int — loops are unrolled
) -> Float[Array, "n"]:
    """Nocedal two-loop recursion returning the L-BFGS search direction -H*g.

    Inactive history slots are masked to no-ops so the function is safe
    to call before the history is fully populated.  When m_used == 0 the
    result is the zero vector; the caller should fall back to -g/||g||.
    """
    alpha_arr = jnp.zeros(m_lbfgs)
    q = g

    for i in range(m_lbfgs):
        buf_idx = (newest - i + m_lbfgs) % m_lbfgs
        active  = i < m_used
        si      = s_buf[buf_idx]
        yi      = y_buf[buf_idx]
        rho_i   = rho_buf[buf_idx]
        alpha_i = rho_i * jnp.dot(si, q)
        alpha_arr = jnp.where(active, alpha_arr.at[buf_idx].set(alpha_i), alpha_arr)
        q = jnp.where(active, q - alpha_i * yi, q)

    # Shanno-Kettler H₀ scaling from the most recent pair
    sy    = jnp.dot(s_buf[newest], y_buf[newest])
    yy    = jnp.dot(y_buf[newest], y_buf[newest])
    gamma = sy / (yy + 1e-18)
    r     = gamma * q

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
# Cost components
# ---------------------------------------------------------------------------

def _smoothness_cost(
    traj:   Float[Array, "T DOF"],
    w_vel:  float,
    w_acc:  float,
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
    jerk  = acc[1:] - acc[:-1]
    cost  = w_acc  * jnp.sum(acc  ** 2)
    cost += w_jerk * jnp.sum(jerk ** 2)
    return cost


def _limits_cost(
    traj:  Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
) -> Array:
    """Squared exceedance penalty for joint-limit violations."""
    viol_upper = jnp.maximum(0.0, traj - upper)
    viol_lower = jnp.maximum(0.0, lower - traj)
    return jnp.sum((viol_upper + viol_lower) ** 2)


def _collision_distances_all(
    cfg:        Float[Array, "DOF"],
    robot:      Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
) -> Float[Array, "P"]:
    """Flat concatenation of all self + world collision distances.

    Each array is ravelled to 1-D first because ``compute_world_collision_distance``
    can return varying-rank tensors depending on geometry type.
    Used only in the final nonlinear evaluation cost.
    """
    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    parts = [jnp.ravel(self_dists)]
    for wg in world_geoms:
        parts.append(jnp.ravel(
            robot_coll.compute_world_collision_distance(robot, cfg, wg)
        ))
    return jnp.concatenate(parts)


def _collision_dists_reduced(
    cfg:         Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    temperature: float,
) -> Float[Array, "G"]:
    """Per-group smooth-minimum collision distances.

    Returns one scalar per collision group (self-collision + one per world
    geometry type), computed as the smooth-minimum over all pair distances in
    that group:

        smooth_min(d) = -temperature * logsumexp(-d / temperature)

    This reduces the Jacobian shape from [P, DOF] (P ≈ 100-300) to
    [G, DOF] (G = 1 + n_world_geoms, typically 3-5), cutting Jacobian
    memory and compile time by 50-100x.
    """
    def smooth_min(d_flat: Array) -> Array:
        return -temperature * jax.scipy.special.logsumexp(-d_flat / temperature)

    self_dists = robot_coll.compute_self_collision_distance(robot, cfg)
    groups = [smooth_min(jnp.ravel(self_dists))]
    for wg in world_geoms:
        dists = robot_coll.compute_world_collision_distance(robot, cfg, wg)
        groups.append(smooth_min(jnp.ravel(dists)))
    return jnp.stack(groups)  # [G]


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def _compute_coll_dists_and_jacs(
    trajs:       Float[Array, "B T DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    temperature: float,
) -> tuple[Float[Array, "B T G"], Float[Array, "B T G DOF"]]:
    """Per-group smooth-min distances and their Jacobians at every waypoint.

    Uses forward-mode AD (``jax.jacfwd``) which requires only DOF=7 JVPs
    instead of the G or P VJPs that reverse-mode would need.  Combined with
    the per-group reduction (G ≈ 4 vs P ≈ 252), this cuts Jacobian memory
    and compile time by roughly two orders of magnitude.

    Returns:
        d_k:  [B, T, G]      — per-group smooth-min distances
        J_k:  [B, T, G, DOF] — ∂d_reduced/∂q at each waypoint
    """
    def per_cfg(cfg):
        d = _collision_dists_reduced(cfg, robot, robot_coll, world_geoms, temperature)
        J = jax.jacfwd(_collision_dists_reduced, argnums=0)(
            cfg, robot, robot_coll, world_geoms, temperature
        )
        return d, J

    d_k, J_k = jax.vmap(jax.vmap(per_cfg))(trajs)
    return d_k, J_k


# ---------------------------------------------------------------------------
# Inner L-BFGS solver (single trajectory)
# ---------------------------------------------------------------------------

def _lbfgs_inner_solve(
    traj:    Float[Array, "T DOF"],
    q_k:     Float[Array, "T DOF"],
    d_k:     Float[Array, "T P"],
    J_k:     Float[Array, "T P DOF"],
    lower:   Float[Array, "DOF"],
    upper:   Float[Array, "DOF"],
    cfg:     TrajOptConfig,
    w_coll:  Array,                    # traced scalar from outer carry
) -> Float[Array, "T DOF"]:
    """Solve the convex inner subproblem for a single trajectory with L-BFGS.

    The inner objective is convex in q because:
      - Smoothness and limits are exact quadratics.
      - The linearized collision hinge max(0, margin - d_lin(q))² is convex
        (hinge of an affine function, squared).
      - The trust-region term ||q - q_k||² is quadratic.

    Start and goal waypoints are pinned by zeroing their gradient components.
    """
    m    = cfg.m_lbfgs       # static Python int — used in range() loops
    T    = traj.shape[0]
    DOF  = traj.shape[1]
    n    = T * DOF

    # Mask that zeros gradients at the first and last waypoints, pinning them.
    endpoint_mask = (
        jnp.ones(n)
        .at[:DOF].set(0.0)
        .at[n - DOF:].set(0.0)
    )

    def cost_fn(x_flat: Float[Array, "n"]) -> Array:
        t = x_flat.reshape(T, DOF)
        c  = cfg.w_smooth * _smoothness_cost(t, cfg.w_vel, cfg.w_acc, cfg.w_jerk)
        c += cfg.w_limits * _limits_cost(t, lower, upper)

        # Linearized collision: d_lin ≈ d_k + J_k @ (q - q_k)
        delta_q = t - q_k                                        # [T, DOF]
        d_lin   = d_k + jnp.einsum("tpd,td->tp", J_k, delta_q) # [T, P]
        violation = jnp.maximum(0.0, cfg.collision_margin - d_lin)
        c += w_coll * jnp.sum(violation ** 2)

        # Trust region
        c += cfg.w_trust * jnp.sum(delta_q ** 2)
        return c

    x0           = traj.reshape(n)
    cost0, g0    = jax.value_and_grad(cost_fn)(x0)
    g0           = g0 * endpoint_mask

    init_carry = (
        x0,                      # current iterate
        x0,                      # best iterate seen so far
        cost0,                   # best cost seen so far
        x0,                      # x_prev  (dummy for iter 0)
        g0,                      # g_prev  (dummy for iter 0)
        jnp.zeros((m, n)),       # s_buf
        jnp.zeros((m, n)),       # y_buf
        jnp.zeros(m),            # rho_buf
        jnp.int32(0),            # m_used
        jnp.int32(0),            # newest
        jnp.int32(0),            # iter_count
    )

    def lbfgs_step(carry, _):
        (x, best_x, best_cost,
         x_prev, g_prev,
         s_buf, y_buf, rho_buf,
         m_used, newest, iter_count) = carry

        cost_val, g = jax.value_and_grad(cost_fn)(x)
        g = g * endpoint_mask

        # --- Update L-BFGS curvature history ---
        s_k   = x - x_prev
        y_k   = g - g_prev
        sy    = jnp.dot(s_k, y_k)
        yy    = jnp.dot(y_k, y_k)
        valid = (sy > 1e-10 * yy + 1e-30) & (iter_count > 0)

        new_newest    = (newest + 1) % m
        actual_newest = jnp.where(valid, new_newest, newest)
        s_buf   = jnp.where(valid, s_buf.at[new_newest].set(s_k),            s_buf)
        y_buf   = jnp.where(valid, y_buf.at[new_newest].set(y_k),            y_buf)
        rho_buf = jnp.where(valid, rho_buf.at[new_newest].set(1.0 / (sy + 1e-30)), rho_buf)
        m_used  = jnp.where(valid & (m_used < m), m_used + 1, m_used)
        newest  = actual_newest

        # --- L-BFGS search direction ---
        dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
        dir_gd    = -g / (jnp.linalg.norm(g) + 1e-18)
        direction = jnp.where(m_used > 0, dir_lbfgs, dir_gd)
        direction = direction * endpoint_mask   # keep endpoints pinned

        # --- 5-point line search ---
        suff_thresh = cost_val * (1.0 - 1e-4)
        trial_costs = jax.vmap(lambda a: cost_fn(x + a * direction))(_LS_ALPHAS)
        has_suff    = trial_costs < suff_thresh
        best_idx    = jnp.where(
            jnp.any(has_suff),
            jnp.argmax(has_suff),
            jnp.argmin(trial_costs),
        )
        alpha   = _LS_ALPHAS[best_idx]
        x_new   = x + alpha * direction
        new_cost = trial_costs[best_idx]

        improved  = new_cost < best_cost
        best_x    = jnp.where(improved, x_new, best_x)
        best_cost = jnp.where(improved, new_cost, best_cost)

        return (
            x_new, best_x, best_cost,
            x, g,
            s_buf, y_buf, rho_buf,
            m_used, newest, iter_count + 1,
        ), None

    (_, best_x, _, _, _, _, _, _, _, _, _), _ = jax.lax.scan(
        lbfgs_step, init_carry, None, length=cfg.n_inner_iters,
    )

    return best_x.reshape(T, DOF)


# ---------------------------------------------------------------------------
# Final nonlinear cost  (used only for ranking at the end)
# ---------------------------------------------------------------------------

def _eval_cost(
    traj:       Float[Array, "T DOF"],
    lower:      Float[Array, "DOF"],
    upper:      Float[Array, "DOF"],
    robot:      Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    cfg:        TrajOptConfig,
) -> Array:
    """Full nonlinear cost at the final w_collision_max weight."""
    cost = cfg.w_smooth * _smoothness_cost(traj, cfg.w_vel, cfg.w_acc, cfg.w_jerk)
    cost += cfg.w_limits * _limits_cost(traj, lower, upper)

    def per_step(c):
        dists = _collision_distances_all(c, robot, robot_coll, world_geoms)
        return jnp.sum(-jnp.minimum(colldist_from_sdf(dists, cfg.collision_margin), 0.0))

    cost += cfg.w_collision_max * jnp.sum(jax.vmap(per_step)(traj))
    return cost


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("opt_cfg",),
)
def sco_trajopt(
    init_trajs:  Float[Array, "B T DOF"],
    start:       Float[Array, "DOF"],
    goal:        Float[Array, "DOF"],
    robot:       Robot,
    robot_coll:  RobotCollision,
    world_geoms: tuple,
    opt_cfg:     TrajOptConfig = TrajOptConfig(),
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """True SCO trajectory optimization.

    Outer loop: linearize collision at current trajectory, solve convex
    inner subproblem with L-BFGS, repeat with scaled-up penalty.

    Args:
        init_trajs:  Initial trajectory batch.  Shape [B, T, DOF].
        start:       Start joint configuration.  Shape [DOF].
        goal:        Goal joint configuration.   Shape [DOF].
        robot:       Robot kinematics pytree.
        robot_coll:  Robot collision model pytree.
        world_geoms: Tuple of stacked world collision geometry objects.
        opt_cfg:     Hyper-parameters (static — changes trigger recompilation).

    Returns:
        best_traj:   Trajectory with lowest final nonlinear cost. [T, DOF].
        costs:       Final nonlinear cost per trajectory.         [B].
        final_trajs: All optimized trajectories.                  [B, T, DOF].
    """
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits

    # Pin endpoints in the initial batch
    trajs = init_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)

    def outer_step(carry, _):
        trajs, w_coll = carry

        # Step 1: linearize collision at the current trajectory
        d_k, J_k = _compute_coll_dists_and_jacs(
            trajs, robot, robot_coll, world_geoms, opt_cfg.smooth_min_temperature
        )

        # Step 2: solve the convex inner subproblem for every trajectory in parallel
        new_trajs = jax.vmap(
            lambda traj, dk, Jk: _lbfgs_inner_solve(
                traj, traj, dk, Jk, lower, upper, opt_cfg, w_coll
            )
        )(trajs, d_k, J_k)

        # Re-pin endpoints (numerical safety)
        new_trajs = new_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)

        # Step 3: scale up the collision penalty for the next outer iteration
        w_coll = jnp.minimum(w_coll * opt_cfg.penalty_scale, opt_cfg.w_collision_max)

        return (new_trajs, w_coll), None

    (final_trajs, _), _ = jax.lax.scan(
        outer_step,
        (trajs, jnp.array(opt_cfg.w_collision, dtype=jnp.float32)),
        None,
        length=opt_cfg.n_outer_iters,
    )

    # Rank by full nonlinear cost at the maximum collision weight
    costs = jax.vmap(
        lambda t: _eval_cost(t, lower, upper, robot, robot_coll, world_geoms, opt_cfg)
    )(final_trajs)
    best_idx  = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]

    return best_traj, costs, final_trajs


# ---------------------------------------------------------------------------
# Convenience: initialise trajectory batch by interpolation
# ---------------------------------------------------------------------------

def make_init_trajs(
    start:       Float[Array, "DOF"],
    goal:        Float[Array, "DOF"],
    n_batch:     int,
    n_timesteps: int,
    key:         Array,
    noise_scale: float = 0.05,
) -> Float[Array, "B T DOF"]:
    """Create a batch of linearly-interpolated trajectories with small random noise."""
    t    = jnp.linspace(0.0, 1.0, n_timesteps)[:, None]
    base = start[None, :] * (1.0 - t) + goal[None, :] * t
    trajs = jnp.broadcast_to(base[None], (n_batch, n_timesteps, start.shape[0]))
    noise = jax.random.normal(key, trajs.shape) * noise_scale
    return trajs + noise
