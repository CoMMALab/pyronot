"""STOMP/MPPI TrajOpt: Sampling-based trajectory optimization.

Algorithm (STOMP/MPPI hybrid):
  At each iteration:
    1. Sample K noisy perturbations δq_k around the current trajectory.
       δq_k ~ N(0, σ² I), interior waypoints only (endpoints zeroed).
    2. Form perturbed trajectories q_k = q + δq_k (endpoints pinned to
       start/goal).
    3. Evaluate the full nonlinear cost for each q_k:
           J(q_k) = w_smooth * J_smooth(q_k)
                  + w_coll   * Σ_t max(0, margin - d(q_k_t))²
                  + w_limits * J_limits(q_k)
    4. Compute importance weights:
           w_k = softmax(-(J_k - min_k J_k) / λ)
    5. Compute the weighted update:  Δq = Σ_k w_k * δq_k.
    6. Optionally apply covariant smoothing (CHOMP metric):
           Δq ← M_int^{-1} Δq   (interior waypoints only).
    7. Apply: q ← q + step_size * Δq, pin endpoints.
    8. Scale w_coll by collision_penalty_scale (penalty continuation).

References:
  Ratliff et al. 2009, "STOMP: Stochastic Trajectory Optimization for
    Motion Planning".
  Williams et al. 2016, "Aggressive Driving with Model Predictive Path
    Integral Control".
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
from ._ik_primitives import _LS_ALPHAS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StompTrajOptConfig:
    """Hyper-parameters for STOMP/MPPI trajectory optimization."""

    # --- Main loop ---
    n_iters: int = 50
    """Number of STOMP/MPPI update iterations."""

    n_samples: int = 128
    """Number of noise samples K per iteration (parallelised via vmap).
    CUDA path supports up to 1024 (max threads per block).  Higher K improves
    GPU occupancy and per-iteration estimate quality.  cuRobo uses 500+."""

    noise_scale: float = 0.05
    """Std-dev σ of isotropic Gaussian perturbations (radians)."""

    temperature: float = 0.1
    """Importance-weighting temperature λ.  Smaller → more exploitative."""

    step_size: float = 0.3
    """EMA-style blend factor applied to the weighted-mean update Δq.
    Equivalent to step_size_mean in cuRobo: new_traj = traj + step_size * delta,
    which equals (1 - step_size) * traj + step_size * weighted_mean.
    Values in [0.1, 0.5] prevent oscillation; 1.0 = full jump (old default)."""

    # --- Smoothness cost weights ---
    w_smooth: float = 1.0
    """Overall smoothness weight."""

    w_vel: float = 1.0
    """Unused; kept for API compatibility."""

    w_acc: float = 0.5
    """Relative weight of acceleration within smoothness."""

    w_jerk: float = 0.1
    """Relative weight of jerk within smoothness."""

    # --- Collision ---
    w_collision: float = 10.0
    """Initial collision penalty weight."""

    w_collision_max: float = 100.0
    """Maximum collision penalty weight after continuation."""

    collision_penalty_scale: float = 1.05
    """Per-iteration multiplicative scaling for w_collision."""

    collision_margin: float = 0.01
    """Activation margin for collision hinge loss (metres)."""

    # --- Joint limits ---
    w_limits: float = 1.0
    """Weight for soft joint-limit violation penalty."""

    # --- Smooth noise (recommended) ---
    use_smooth_noise: bool = True
    """If True, sample perturbations from N(0, noise_scale² · M⁻¹) — the
    smoothness covariance prior — instead of isotropic N(0, σ²I).

    This is the correct STOMP formulation (Ratliff et al. 2009).  With smooth
    noise every sample is a smooth trajectory, so:
      - smoothness cost variation between samples is small;
      - importance weights are driven by collision / limits cost;
      - the weighted update is smooth, preventing roughness accumulation.

    With isotropic noise each step adds ~2·noise_scale² to the velocity
    variance at every timestep, so after N iters the trajectory becomes
    N·DOF·2·noise_scale² rougher — completely defeating the optimizer.
    """

    # --- Covariant update (optional, legacy) ---
    use_covariant_update: bool = False
    """If True, post-multiply the weighted-noise update by M⁻¹.

    Only makes sense when use_smooth_noise=False.  When use_smooth_noise=True
    a covariant update is redundant (the noise already respects the metric).
    When use_smooth_noise=False and this is True, M⁻¹ amplifies smooth
    components by up to 1/smoothness_reg, causing divergence — leave False.
    """

    smoothness_reg: float = 1e-3
    """Diagonal regulariser added before factoring the smoothness metric."""

    use_cost_normalization: bool = True
    """If True, normalize cost differences by std(costs) before the softmax.

    Makes temperature scale-invariant: temperature=1.0 means a 1-std cost
    difference gives weight ratio exp(-1) ≈ 0.37.  Without normalization,
    absolute costs of 100-1000 with temperature=0.1 cause exp(-1000/0.1)≈0,
    collapsing all weight onto a single sample and turning STOMP into a
    random walk.

    Mirrors the implicit scale handling in cuRobo's beta parameter."""

    use_null_particle: bool = True
    """If True, replace sample k=0 with the zero-noise (current trajectory).

    Guarantees the update cannot move away from the current trajectory when
    all noisy samples are worse: the null particle then dominates the softmax
    weights, driving delta → 0.  This provides implicit best-trajectory
    stabilisation without a separate memory buffer.

    Equivalent to cuRobo's null_per_problem safeguard."""

    use_elite_filter: bool = True
    """If True, keep only the lowest-cost elite fraction before softmax."""

    elite_frac: float = 0.25
    """Fraction of samples retained for elite weighting (CEM/MPPI hybrid)."""

    adaptive_covariance: bool = True
    """If True, adapt perturbation scale from weighted sample variance."""

    cov_update_rate: float = 0.2
    """EMA rate for covariance/scale adaptation (cuRobo step_size_cov-like)."""

    noise_decay: float = 0.99
    """Per-iteration multiplicative decay for perturbation scale."""

    noise_scale_min: float = 0.005
    """Lower bound for adaptive perturbation scale."""

    noise_scale_max: float = 0.2
    """Upper bound for adaptive perturbation scale."""

    normalize_smooth_noise_scale: bool = True
    """If True, rescale smooth-noise sigma so average waypoint std matches
    ``noise_scale`` instead of being amplified by ``M^{-1}``."""

    n_lbfgs_iters: int = 10
    """Post-MPPI L-BFGS refinement iterations (0 disables Stage 2)."""

    m_lbfgs: int = 5
    """L-BFGS history size for the post-MPPI refinement stage."""

    lbfgs_step_scale: float = 1.0
    """Global multiplier on line-search alpha candidates for Stage 2."""

    track_best_trajectory: bool = True
    """If True, keep the best sampled trajectory across MPPI iterations.

    This mirrors cuRobo's best-seed retention behavior and prevents late
    iterations from drifting away from an earlier better candidate.
    """


# ---------------------------------------------------------------------------
# Cost components  (same as CHOMP for API parity)
# ---------------------------------------------------------------------------

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
    jerk  = acc[1:] - acc[:-1]
    cost  = w_acc  * jnp.sum(acc  ** 2)
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
    """Flat concatenation of all self + world signed distances."""
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


# ---------------------------------------------------------------------------
# Covariant-update preconditioning  (shared with CHOMP)
# ---------------------------------------------------------------------------

def _build_stomp_metric(
    n_timesteps: int,
    reg: float,
) -> Float[Array, "Tin Tin"]:
    """Interior smoothness precision matrix M = D^T D + reg·I (second-difference)."""
    n = n_timesteps
    if n <= 2:
        return jnp.eye(1, dtype=jnp.float32)

    D = jnp.zeros((n - 2, n), dtype=jnp.float32)
    rows = jnp.arange(n - 2)
    D = D.at[rows, rows].set(1.0)
    D = D.at[rows, rows + 1].set(-2.0)
    D = D.at[rows, rows + 2].set(1.0)
    R = D.T @ D
    R_interior = R[1:-1, 1:-1]
    return R_interior + reg * jnp.eye(n - 2, dtype=jnp.float32)


def _build_stomp_chol(
    n_timesteps: int,
    reg: float,
) -> tuple[Float[Array, "Tin Tin"], Float[Array, "Tin Tin"]]:
    """Return (M, L) where M is the interior smoothness metric and L = cholesky(M).

    Sampling z ~ N(0, I) and solving L^T eps = noise_scale·z gives
    eps ~ N(0, noise_scale²·M⁻¹) — smooth perturbations that respect the
    trajectory smoothness prior.
    """
    M = _build_stomp_metric(n_timesteps, reg)
    L = jnp.linalg.cholesky(M)  # lower triangular, M = L L^T
    return M, L


def _apply_covariant_update(
    delta: Float[Array, "T DOF"],
    metric: Float[Array, "Tin Tin"],
) -> Float[Array, "T DOF"]:
    """Precondition interior waypoints of delta with the smoothness metric."""
    if delta.shape[0] <= 2:
        return delta

    interior = delta[1:-1]  # [Tin, DOF]
    smooth_one = lambda g: jnp.linalg.solve(metric, g)
    interior_smooth = jax.vmap(smooth_one, in_axes=1, out_axes=1)(interior)

    return jnp.concatenate([
        delta[:1],
        interior_smooth,
        delta[-1:],
    ], axis=0)


def _smooth_noise_gain(L: Float[Array, "Tin Tin"]) -> Array:
    """RMS gain from z to eps for eps = L^{-T} z (scalar)."""
    Tin = L.shape[0]
    if Tin == 0:
        return jnp.array(1.0, dtype=jnp.float32)
    eye = jnp.eye(Tin, dtype=jnp.float32)
    # Solve L^T X = I  => X = L^{-T}
    L_inv_T = jax.scipy.linalg.solve_triangular(
        L, eye, lower=True, trans="T"
    )
    row_var = jnp.sum(L_inv_T * L_inv_T, axis=1)
    return jnp.sqrt(jnp.maximum(jnp.mean(row_var), 1e-12))


# Number of taps and sigma for the 1D Gaussian smoothing kernel used as a
# fast replacement for the O(T²) Cholesky-based smooth-noise generation.
_SMOOTH_KERNEL_HALF_WIDTH = 3
_SMOOTH_KERNEL_SIGMA = 2.0


def _build_gauss_kernel() -> Float[Array, "W"]:
    """Build a normalised 1D Gaussian smoothing kernel."""
    hw = _SMOOTH_KERNEL_HALF_WIDTH
    x = jnp.arange(-hw, hw + 1, dtype=jnp.float32)
    k = jnp.exp(-0.5 * (x / _SMOOTH_KERNEL_SIGMA) ** 2)
    return k / jnp.sum(k)


def _gauss_conv_gain() -> Array:
    """RMS gain of the Gaussian convolution kernel (scalar)."""
    k = _build_gauss_kernel()
    return jnp.sqrt(jnp.sum(k ** 2))


def _lbfgs_two_loop(
    g: Float[Array, "n"],
    s_buf: Float[Array, "m n"],
    y_buf: Float[Array, "m n"],
    rho_buf: Float[Array, "m"],
    m_used: Array,
    newest: Array,
    m_lbfgs: int,
) -> Float[Array, "n"]:
    """Nocedal two-loop recursion returning -H*g."""
    alpha_arr = jnp.zeros(m_lbfgs, dtype=jnp.float32)
    q = g

    for i in range(m_lbfgs):
        buf_idx = (newest - i + m_lbfgs) % m_lbfgs
        active = i < m_used
        si = s_buf[buf_idx]
        yi = y_buf[buf_idx]
        rho_i = rho_buf[buf_idx]
        alpha_i = rho_i * jnp.dot(si, q)
        alpha_arr = jnp.where(active, alpha_arr.at[buf_idx].set(alpha_i), alpha_arr)
        q = jnp.where(active, q - alpha_i * yi, q)

    sy = jnp.dot(s_buf[newest], y_buf[newest])
    yy = jnp.dot(y_buf[newest], y_buf[newest])
    gamma = sy / (yy + 1e-18)
    r = gamma * q

    for step in range(m_lbfgs):
        buf_idx = (newest - m_used + 1 + step + m_lbfgs) % m_lbfgs
        active = step < m_used
        si = s_buf[buf_idx]
        yi = y_buf[buf_idx]
        rho_i = rho_buf[buf_idx]
        alpha_i = alpha_arr[buf_idx]
        beta = rho_i * jnp.dot(yi, r)
        r = jnp.where(active, r + si * (alpha_i - beta), r)

    return -r


def _lbfgs_refine_single(
    init_traj: Float[Array, "T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: StompTrajOptConfig,
) -> Float[Array, "T DOF"]:
    """Stage 2 refinement: L-BFGS on final nonlinear objective."""
    T, DOF = init_traj.shape
    n = T * DOF
    m = opt_cfg.m_lbfgs

    endpoint_mask = (
        jnp.ones((T, DOF), dtype=jnp.float32).at[0, :].set(0.0).at[-1, :].set(0.0)
    )
    lower_traj = jnp.broadcast_to(lower[None, :].astype(jnp.float32), (T, DOF))
    upper_traj = jnp.broadcast_to(upper[None, :].astype(jnp.float32), (T, DOF))
    start = start.astype(jnp.float32)
    goal  = goal.astype(jnp.float32)

    def _pin_and_clip(traj: Float[Array, "T DOF"]) -> Float[Array, "T DOF"]:
        traj = jnp.clip(traj, lower_traj, upper_traj)
        return traj.at[0, :].set(start).at[-1, :].set(goal)

    def cost_from_flat(x_flat: Float[Array, "n"]) -> Array:
        traj = _pin_and_clip(x_flat.reshape(T, DOF))
        return _eval_cost(traj, lower, upper, robot, robot_coll, world_geoms, opt_cfg)

    x0 = _pin_and_clip(init_traj).reshape(-1).astype(jnp.float32)
    g0 = jax.grad(cost_from_flat)(x0).reshape(T, DOF) * endpoint_mask
    g0 = g0.reshape(-1).astype(jnp.float32)
    c0 = cost_from_flat(x0).astype(jnp.float32)

    init_carry = (
        x0, x0, c0, x0, jnp.zeros(n, dtype=jnp.float32),
        jnp.zeros((m, n), dtype=jnp.float32),
        jnp.zeros((m, n), dtype=jnp.float32),
        jnp.zeros((m,), dtype=jnp.float32),
        jnp.int32(0), jnp.int32(0),
        g0,
    )

    def body(carry, _):
        x, best_x, best_cost, x_prev, g_prev, s_buf, y_buf, rho_buf, m_used, newest, _ = carry

        cost = cost_from_flat(x).astype(jnp.float32)
        g = jax.grad(cost_from_flat)(x).reshape(T, DOF) * endpoint_mask
        g = g.reshape(-1).astype(jnp.float32)

        s_k = x - x_prev
        y_k = g - g_prev
        sy = jnp.dot(s_k, y_k)
        yy = jnp.dot(y_k, y_k)
        valid = (sy > 1e-10 * yy + 1e-30) & (m_used >= 0)

        new_newest = (newest + 1) % m
        actual_newest = jnp.where(valid, new_newest, newest)
        s_buf = jnp.where(valid, s_buf.at[new_newest].set(s_k), s_buf)
        y_buf = jnp.where(valid, y_buf.at[new_newest].set(y_k), y_buf)
        rho_buf = jnp.where(valid, rho_buf.at[new_newest].set(1.0 / (sy + 1e-30)), rho_buf)
        m_used = jnp.where(valid & (m_used < m), m_used + 1, m_used)
        newest = actual_newest

        dir_lbfgs = _lbfgs_two_loop(g, s_buf, y_buf, rho_buf, m_used, newest, m)
        dir_gd = -g / (jnp.linalg.norm(g) + 1e-18)
        direction = jnp.where(m_used > 0, dir_lbfgs, dir_gd)
        direction = (direction.reshape(T, DOF) * endpoint_mask).reshape(-1)

        alphas = (opt_cfg.lbfgs_step_scale * _LS_ALPHAS).astype(jnp.float32)

        def eval_alpha(a):
            x_try = x + a * direction
            return cost_from_flat(x_try)

        trial_costs = jax.vmap(eval_alpha)(alphas).astype(jnp.float32)
        suff_thresh = cost * jnp.float32(1.0 - 1e-4)
        has_suff = trial_costs < suff_thresh
        best_idx = jnp.where(jnp.any(has_suff), jnp.argmax(has_suff), jnp.argmin(trial_costs))
        x_new = x + alphas[best_idx] * direction
        step_cost = trial_costs[best_idx]

        improved = step_cost < best_cost
        best_x = jnp.where(improved, x_new, best_x)
        best_cost = jnp.where(improved, step_cost, best_cost)

        return (
            x_new, best_x, best_cost, x, g, s_buf, y_buf, rho_buf, m_used, newest, g
        ), None

    (_, best_x, _, _, _, _, _, _, _, _, _), _ = jax.lax.scan(
        body, init_carry, None, length=opt_cfg.n_lbfgs_iters
    )
    return _pin_and_clip(best_x.reshape(T, DOF))


# ---------------------------------------------------------------------------
# Single-trajectory STOMP optimizer
# ---------------------------------------------------------------------------

def _optimize_single_traj(
    init_traj: Float[Array, "T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    metric: Float[Array, "Tin Tin"],
    L: Float[Array, "Tin Tin"],
    opt_cfg: StompTrajOptConfig,
    key: Array,
) -> Float[Array, "T DOF"]:
    """Run Stage 1 (STOMP/MPPI) iterations on one trajectory."""
    T, DOF = init_traj.shape
    T_in = T - 2  # number of interior (non-endpoint) timesteps
    traj = init_traj.at[0].set(start).at[-1].set(goal)
    K = opt_cfg.n_samples
    n_elite = max(1, min(K, int(round(opt_cfg.elite_frac * K))))

    def cost_fn(t: Float[Array, "T DOF"], w_coll: Array) -> Array:
        c = opt_cfg.w_smooth * _smoothness_cost(
            t, opt_cfg.w_vel, opt_cfg.w_acc, opt_cfg.w_jerk
        )
        c += opt_cfg.w_limits * _limits_cost(t, lower, upper)
        c += w_coll * _collision_cost(
            t, robot, robot_coll, world_geoms, opt_cfg.collision_margin
        )
        return c

    w_coll0 = jnp.array(opt_cfg.w_collision, dtype=jnp.float32)

    # Isotropic-noise mask (only used when use_smooth_noise=False)
    endpoint_mask = (
        jnp.ones((T, DOF), dtype=jnp.float32)
        .at[0, :].set(0.0)
        .at[-1, :].set(0.0)
    )
    lower_t = lower[None, :].astype(jnp.float32)
    upper_t = upper[None, :].astype(jnp.float32)

    # Per-DOF sigma vector — allows each joint to adapt its own perturbation scale.
    if opt_cfg.use_smooth_noise and T_in > 0 and opt_cfg.normalize_smooth_noise_scale:
        sigma_scalar = opt_cfg.noise_scale / _gauss_conv_gain()
    else:
        sigma_scalar = opt_cfg.noise_scale
    sigma0 = jnp.full((DOF,), sigma_scalar, dtype=jnp.float32)
    sigma0 = jnp.clip(
        sigma0,
        jnp.array(opt_cfg.noise_scale_min, dtype=jnp.float32),
        jnp.array(opt_cfg.noise_scale_max, dtype=jnp.float32),
    )

    # Precompute the Gaussian smoothing kernel (used inside scan body).
    gauss_kernel = _build_gauss_kernel()       # [W]
    pad_w = _SMOOTH_KERNEL_HALF_WIDTH

    def step_fn(carry, _):
        curr_traj, best_traj, best_cost, w_coll, sigma_curr, step_key = carry
        step_key, sample_key = jax.random.split(step_key)

        # --- Sample K noise perturbations ---
        if opt_cfg.use_smooth_noise and T_in > 0:
            # Fast smooth noise via 1D Gaussian convolution — O(K·T·DOF·W)
            # instead of O(T²·K·DOF) for the Cholesky triangular solve.
            # sigma_curr is [DOF], broadcast over (K, T_in) dims.
            z = (jax.random.normal(sample_key, (K, T_in, DOF), dtype=jnp.float32)
                 * sigma_curr[None, None, :])              # [K, T_in, DOF]
            # Reshape to (K*DOF, 1, T_in) for batched 1-D convolution
            z_conv = z.transpose(0, 2, 1).reshape(K * DOF, 1, T_in)
            kernel_1d = gauss_kernel[None, None, :]         # [1, 1, W]
            eps_conv = jax.lax.conv_general_dilated(
                z_conv, kernel_1d,
                window_strides=(1,),
                padding=((pad_w, pad_w),),
            )                                               # [K*DOF, 1, T_in]
            eps_interior = (
                eps_conv[:, 0, :]                           # [K*DOF, T_in]
                .reshape(K, DOF, T_in)
                .transpose(0, 2, 1)                         # [K, T_in, DOF]
            )
            pad = jnp.zeros((K, 1, DOF), dtype=jnp.float32)
            noise = jnp.concatenate([pad, eps_interior, pad], axis=1)  # [K, T, DOF]
        else:
            # Isotropic noise (MPPI-style), endpoints zeroed
            noise = (
                jax.random.normal(sample_key, (K, T, DOF), dtype=jnp.float32)
                * sigma_curr[None, None, :]
                * endpoint_mask[None]
            )

        # --- Null particle: k=0 evaluates the current trajectory (no noise) ---
        # If the current trajectory is better than all noisy samples its weight
        # dominates and delta → 0, preventing divergence (implicit best-traj
        # stabilisation without a separate memory buffer).
        if opt_cfg.use_null_particle:
            noise = noise.at[0].set(0.0)

        # --- Evaluate cost of each perturbed trajectory ---
        perturbed = curr_traj[None] + noise        # [K, T, DOF]
        perturbed = perturbed.at[:, 0, :].set(start).at[:, -1, :].set(goal)
        perturbed = jnp.clip(perturbed, lower_t[None], upper_t[None])
        perturbed = perturbed.at[:, 0, :].set(start).at[:, -1, :].set(goal)
        costs = jax.vmap(lambda t: cost_fn(t, w_coll))(perturbed).astype(jnp.float32)  # [K]

        # --- Best-candidate tracking (cuRobo-like) ---
        min_idx = jnp.argmin(costs)
        iter_best_cost = costs[min_idx]
        iter_best_traj = perturbed[min_idx]
        improved = iter_best_cost < best_cost
        best_traj = jnp.where(improved, iter_best_traj, best_traj)
        best_cost = jnp.where(improved, iter_best_cost, best_cost)

        # --- Importance weights: scale-invariant softmax ---
        # Normalize shifted costs by their std so that `temperature` is
        # dimensionless (1.0 → exp(-1) ≈ 0.37 weight ratio per 1-std gap).
        # Without this, absolute costs of 100–1000 with temperature=0.1 drive
        # exp(-cost/0.1) → 0 for all but the single best sample.
        if opt_cfg.use_elite_filter and n_elite < K:
            _, elite_idx = jax.lax.top_k(-costs, n_elite)
            elite_mask = jnp.zeros((K,), dtype=bool).at[elite_idx].set(True)
        else:
            elite_mask = jnp.ones((K,), dtype=bool)
        active_costs = jnp.where(elite_mask, costs, jnp.inf)
        min_active = jnp.min(active_costs)
        costs_shifted = jnp.where(elite_mask, costs - min_active, jnp.inf)
        if opt_cfg.use_cost_normalization:
            finite_shift = jnp.where(elite_mask, costs_shifted, 0.0)
            n_active = jnp.maximum(jnp.sum(elite_mask), 1)
            mean_shift = jnp.sum(finite_shift) / n_active
            var_shift = jnp.sum(elite_mask * (finite_shift - mean_shift) ** 2) / n_active
            beta = jnp.maximum(jnp.sqrt(var_shift), 1e-6) * opt_cfg.temperature
        else:
            beta = opt_cfg.temperature
        log_w = jnp.where(elite_mask, -costs_shifted / (beta + 1e-18), -jnp.inf)
        weights = jax.nn.softmax(log_w)            # [K]

        # --- Weighted update: Δq = Σ_k w_k · δq_k ---
        delta = jnp.einsum("k,ktd->td", weights, noise)  # [T, DOF]

        # --- Optional covariant smoothing (legacy, use_smooth_noise=False only) ---
        if opt_cfg.use_covariant_update:
            delta = _apply_covariant_update(delta, metric)

        # --- Apply update (step_size acts as EMA blend factor α) ---
        next_traj = curr_traj + opt_cfg.step_size * delta
        next_traj = jnp.clip(next_traj, lower_t, upper_t)
        next_traj = next_traj.at[0, :].set(start).at[-1, :].set(goal)

        # --- Adaptive covariance / noise schedule (per-DOF, cuRobo-like) ---
        if opt_cfg.adaptive_covariance:
            interior_noise = noise[:, 1:-1, :] if T_in > 0 else noise
            # Per-DOF weighted variance: var_d = sum_k w_k * mean_t(noise_k_t_d^2)
            # interior_noise: [K, T_in, DOF] → per-sample per-DOF mean variance [K, DOF]
            var_k_d = jnp.mean(interior_noise ** 2, axis=1)   # [K, DOF]
            target_sigma = jnp.sqrt(
                jnp.maximum(jnp.einsum("k,kd->d", weights, var_k_d), 1e-12)
            )                                                   # [DOF]
            sigma_next = (
                (1.0 - opt_cfg.cov_update_rate) * sigma_curr
                + opt_cfg.cov_update_rate * target_sigma
            )
        else:
            sigma_next = sigma_curr
        sigma_next = sigma_next * opt_cfg.noise_decay
        sigma_next = jnp.clip(
            sigma_next,
            jnp.array(opt_cfg.noise_scale_min, dtype=jnp.float32),
            jnp.array(opt_cfg.noise_scale_max, dtype=jnp.float32),
        )

        # --- Scale collision penalty (continuation) ---
        w_coll = jnp.minimum(
            w_coll * opt_cfg.collision_penalty_scale,
            jnp.array(opt_cfg.w_collision_max, dtype=jnp.float32),
        )

        return (next_traj, best_traj, best_cost, w_coll, sigma_next, step_key), None

    init_best_cost = cost_fn(traj, w_coll0).astype(jnp.float32)
    (final_traj, best_traj, _, _, _, _), _ = jax.lax.scan(
        step_fn,
        (traj, traj, init_best_cost, w_coll0, sigma0, key),
        None,
        length=opt_cfg.n_iters,
    )
    return jnp.where(opt_cfg.track_best_trajectory, best_traj, final_traj)


# ---------------------------------------------------------------------------
# Final cost evaluation for ranking
# ---------------------------------------------------------------------------

def _eval_cost(
    traj: Float[Array, "T DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: StompTrajOptConfig,
) -> Array:
    """Full nonlinear cost at w_collision_max for trajectory ranking."""
    c = opt_cfg.w_smooth * _smoothness_cost(
        traj, opt_cfg.w_vel, opt_cfg.w_acc, opt_cfg.w_jerk
    )
    c += opt_cfg.w_limits * _limits_cost(traj, lower, upper)
    c += opt_cfg.w_collision_max * _collision_cost(
        traj, robot, robot_coll, world_geoms, opt_cfg.collision_margin
    )
    return c


def _lbfgs_refine_batch(
    trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    lower: Float[Array, "DOF"],
    upper: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: StompTrajOptConfig,
) -> Float[Array, "B T DOF"]:
    if opt_cfg.n_lbfgs_iters <= 0:
        return trajs
    return jax.vmap(
        lambda t: _lbfgs_refine_single(
            t, start, goal, lower, upper, robot, robot_coll, world_geoms, opt_cfg
        )
    )(trajs)


# ---------------------------------------------------------------------------
# Batched JIT-compiled entrypoint
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("opt_cfg",))
def _stomp_trajopt_jax(
    init_trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    key: Array,
    opt_cfg: StompTrajOptConfig = StompTrajOptConfig(),
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    lower = robot.joints.lower_limits.astype(jnp.float32)
    upper = robot.joints.upper_limits.astype(jnp.float32)

    metric, L = _build_stomp_chol(init_trajs.shape[1], opt_cfg.smoothness_reg)

    # Give each trajectory in the batch a different PRNG key.
    B = init_trajs.shape[0]
    keys = jax.random.split(key, B)

    final_trajs = jax.vmap(
        lambda t, k: _optimize_single_traj(
            t, start, goal, lower, upper, robot, robot_coll,
            world_geoms, metric, L, opt_cfg, k,
        )
    )(init_trajs, keys)

    final_trajs = _lbfgs_refine_batch(
        final_trajs, start, goal, lower, upper, robot, robot_coll, world_geoms, opt_cfg
    )
    final_trajs = final_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)

    costs = jax.vmap(
        lambda t: _eval_cost(t, lower, upper, robot, robot_coll, world_geoms, opt_cfg)
    )(final_trajs)
    best_idx = jnp.argmin(costs)
    best_traj = final_trajs[best_idx]
    return best_traj, costs, final_trajs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stomp_trajopt(
    init_trajs: Float[Array, "B T DOF"],
    start: Float[Array, "DOF"],
    goal: Float[Array, "DOF"],
    robot: Robot,
    robot_coll: RobotCollision,
    world_geoms: tuple,
    opt_cfg: StompTrajOptConfig = StompTrajOptConfig(),
    *,
    key: Array | None = None,
    use_cuda: bool = False,
) -> tuple[Float[Array, "T DOF"], Float[Array, "B"], Float[Array, "B T DOF"]]:
    """STOMP/MPPI trajectory optimization.

    Sampling-based trajectory optimizer: samples K noisy perturbations per
    iteration, evaluates the full nonlinear cost for each, then updates the
    trajectory via importance-weighted averaging.

    Args:
        init_trajs:  Initial trajectory batch.  Shape [B, T, DOF].
        start:       Start joint configuration.  Shape [DOF].
        goal:        Goal joint configuration.   Shape [DOF].
        robot:       Robot kinematics pytree.
        robot_coll:  Robot collision model pytree.
        world_geoms: Tuple of stacked world collision geometry objects.
        opt_cfg:     STOMP hyper-parameters (static under JIT).
        key:         JAX PRNG key.  Defaults to PRNGKey(0).
        use_cuda:    If True, run the CUDA kernel (requires compiled .so).

    Returns:
        best_traj:   Trajectory with lowest final cost. [T, DOF].
        costs:       Final cost per trajectory.         [B].
        final_trajs: All optimized trajectories.        [B, T, DOF].
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    lower = robot.joints.lower_limits.astype(jnp.float32)
    upper = robot.joints.upper_limits.astype(jnp.float32)

    if use_cuda:
        from ..cuda_kernels._stomp_trajopt_cuda import stomp_trajopt_cuda
        best_traj, costs, final_trajs = stomp_trajopt_cuda(
            init_trajs, start, goal, robot, robot_coll, world_geoms, opt_cfg, key=key
        )
        if opt_cfg.n_lbfgs_iters > 0:
            final_trajs = _lbfgs_refine_batch(
                final_trajs, start, goal, lower, upper,
                robot, robot_coll, world_geoms, opt_cfg
            )
            final_trajs = final_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)
            costs = jax.vmap(
                lambda t: _eval_cost(t, lower, upper, robot, robot_coll, world_geoms, opt_cfg)
            )(final_trajs)
            best_idx = jnp.argmin(costs)
            best_traj = final_trajs[best_idx]
        return best_traj, costs, final_trajs
    return _stomp_trajopt_jax(
        init_trajs, start, goal, robot, robot_coll, world_geoms, key, opt_cfg
    )
