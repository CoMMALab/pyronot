# SCO TrajOpt: Sequential Convex Optimization for Trajectory Planning

This document explains the theory behind the SCO trajectory optimizer, its implementation in pyronot, and the key engineering decisions made to make it fast under JAX/JIT.

**Reference:** Schulman et al., *"Finding Locally Optimal, Collision-Free Trajectories with Sequential Convex Optimization"*, RSS 2013.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [SCO Algorithm Overview](#2-sco-algorithm-overview)
3. [Cost Components](#3-cost-components)
4. [Collision Linearization and Jacobians](#4-collision-linearization-and-jacobians)
5. [Inner Solver: L-BFGS](#5-inner-solver-l-bfgs)
6. [Penalty Continuation](#6-penalty-continuation)
7. [Dimensionality Reduction via Smooth-Min](#7-dimensionality-reduction-via-smooth-min)
8. [Batched Optimization and Initialization](#8-batched-optimization-and-initialization)
9. [JAX Implementation Details](#9-jax-implementation-details)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Problem Formulation

Given a robot with $n$ degrees of freedom, find a trajectory

$$\mathbf{q} = [q_0, q_1, \ldots, q_{T-1}] \in \mathbb{R}^{T \times n}$$

that minimizes a combination of smoothness and constraint-violation costs, subject to:

- **Fixed endpoints:** $q_0 = q_\text{start}$, $q_{T-1} = q_\text{goal}$
- **Joint limits:** $q_\text{lower} \leq q_t \leq q_\text{upper}$ for all $t$
- **Collision avoidance:** $d(q_t) \geq 0$ for all $t$, where $d$ is the signed distance to the nearest obstacle or self-collision pair

The full nonlinear objective is:

$$\min_{\mathbf{q}} \; w_\text{smooth} \cdot J_\text{smooth}(\mathbf{q}) + w_\text{coll} \cdot J_\text{coll}(\mathbf{q}) + w_\text{limits} \cdot J_\text{limits}(\mathbf{q})$$

Collision avoidance makes this non-convex. SCO handles this by repeatedly **linearizing** the collision distances and solving a convex approximation.

---

## 2. SCO Algorithm Overview

The outer loop runs for `n_outer_iters` iterations. Each outer iteration:

1. **Linearize** collision distances at the current trajectory $\mathbf{q}^k$:

$$d_\text{lin}(q_t) = d(q^k_t) + J_d(q^k_t) \cdot (q_t - q^k_t)$$

where $J_d \in \mathbb{R}^{G \times n}$ is the Jacobian of the (reduced) collision distances with respect to joint angles, computed via forward-mode AD.

2. **Solve** the convex inner subproblem with L-BFGS (`n_inner_iters` steps):

$$\min_{q} \; w_\text{smooth} \cdot J_\text{smooth}(q)
    + w_\text{coll} \cdot \sum_t \sum_g \max\!\left(0,\; m - d_\text{lin}^g(q_t)\right)^2
    + w_\text{trust} \cdot \|q - q^k\|^2
    + w_\text{limits} \cdot J_\text{limits}(q)$$

This is convex because the collision term is a squared hinge on an affine function, and all other terms are quadratic.

3. **Update** $\mathbf{q}^{k+1} \leftarrow$ solution.

4. **Scale** $w_\text{coll} \leftarrow \min(w_\text{coll} \cdot \text{penalty\_scale},\; w_\text{coll}^\text{max})$.

The trust-region term $w_\text{trust} \cdot \|q - q^k\|^2$ keeps the inner solution close to the linearization point, where the linear approximation is most accurate. It is not adapted (no shrink/expand logic); its weight is fixed.

---

## 3. Cost Components

### 3.1 Smoothness Cost

The smoothness cost penalizes acceleration and jerk using a 4th-order central-difference stencil:

$$\text{acc}_t = \frac{-q_{t} + 16 q_{t+1} - 30 q_{t+2} + 16 q_{t+3} - q_{t+4}}{12}$$

$$J_\text{smooth} = w_\text{acc} \sum_t \|\text{acc}_t\|^2 + w_\text{jerk} \sum_t \|\text{acc}_{t+1} - \text{acc}_t\|^2$$

This requires at least 5 waypoints and naturally enforces smooth, robot-friendly motion without explicitly penalizing velocity.

### 3.2 Joint-Limit Cost

Soft squared exceedance penalty:

$$J_\text{limits} = \sum_t \sum_i \left(\max(0, q_{t,i} - q^\text{upper}_i) + \max(0, q^\text{lower}_i - q_{t,i})\right)^2$$

### 3.3 Collision Cost (Linearized, Inner)

Inside the inner L-BFGS solve, the non-convex collision distances are replaced by their first-order Taylor expansion around the current outer iterate $q^k$:

$$J_\text{coll}^\text{lin} = w_\text{coll} \sum_t \sum_g \max\!\left(0,\; m - d_\text{lin}^g(q_t)\right)^2$$

where $m$ is the `collision_margin` (a safety buffer in metres, default 0.01 m) and $g$ indexes collision groups (see Section 7).

### 3.4 Final Nonlinear Cost (for ranking)

After all outer iterations, trajectories are ranked using the full nonlinear collision cost at the maximum penalty weight:

$$J_\text{eval} = w_\text{smooth} \cdot J_\text{smooth} + w_\text{limits} \cdot J_\text{limits} + w_\text{coll}^\text{max} \sum_t \sum_p \max\!\left(0,\; -d^p(q_t)\right)$$

This uses all raw collision distances (not reduced), giving an accurate ranking of the batch.

---

## 4. Collision Linearization and Jacobians

At each outer iteration, the Jacobian $J_d \in \mathbb{R}^{G \times n}$ is computed at every waypoint of every trajectory in the batch.

**Forward-mode AD** is used (`jax.jacfwd`) because the output dimension $G$ (number of collision groups, typically 2–5) is much smaller than the input dimension $n$ (DOF, typically 7). Forward-mode requires $n$ JVPs, while reverse-mode would require $G$ VJPs — for $G \ll n$ the trade-off favours reverse mode, but the key bottleneck here is memory (the Jacobian buffer $[B, T, G, n]$) and compile time, not raw FLOPs. Forward-mode compiles significantly faster in practice.

The function being differentiated maps $q \mapsto d_\text{reduced}(q) \in \mathbb{R}^G$ (see Section 7). The call pattern is:

```
jax.vmap(jax.vmap(per_cfg))(trajs)   # over batch B, then timesteps T
```

where `per_cfg` computes both $d$ and $J$ at a single configuration.

---

## 5. Inner Solver: L-BFGS

The inner convex subproblem is solved with a self-contained L-BFGS implementation using the Nocedal two-loop recursion. It runs for exactly `n_inner_iters` steps (a static loop unrolled by `jax.lax.scan`) and maintains a history of `m_lbfgs` curvature pairs.

### 5.1 Two-Loop Recursion

Given gradient $g$, history buffers $(s_i, y_i, \rho_i)$, the search direction $p = -H g$ is computed as:

**Forward pass (newest → oldest):**
$$\alpha_i = \rho_i \, s_i^\top q, \quad q \leftarrow q - \alpha_i y_i$$

**Initial scaling (Shanno-Kettler):**
$$\gamma = \frac{s_\text{new}^\top y_\text{new}}{y_\text{new}^\top y_\text{new} + \epsilon}, \quad r \leftarrow \gamma \, q$$

**Backward pass (oldest → newest):**
$$\beta_i = \rho_i \, y_i^\top r, \quad r \leftarrow r + s_i (\alpha_i - \beta_i)$$

The curvature update $\rho_i = 1 / (s_i^\top y_i)$ is only accepted when $s^\top y > 10^{-10} \|y\|^2$ (positive-definite curvature condition), keeping the Hessian estimate well-conditioned.

### 5.2 Line Search

A static 5-point backtracking search is used with step sizes $[1.0, 0.5, 0.25, 0.1, 0.025]$. The best step satisfying a sufficient-decrease condition (relative 1e-4 improvement) is chosen; if none qualifies, the smallest-cost trial is used. All five cost evaluations run in parallel via `jax.vmap`.

### 5.3 Endpoint Pinning

Start and goal waypoints are held fixed throughout the inner solve by multiplying the gradient by an **endpoint mask** that zeros out the first and last DOF-block entries before each L-BFGS update and before each step is applied. This is simpler and JIT-friendly compared to constrained optimization.

### 5.4 Best-Iterate Tracking

The inner solver tracks the best iterate seen across all `n_inner_iters` steps (lowest cost), not just the final iterate. This protects against occasional oscillation near the end of the L-BFGS run.

---

## 6. Penalty Continuation

The collision weight starts at `w_collision` (default 1.0) and is multiplied by `penalty_scale` (default 3.0) after each outer iteration, capped at `w_collision_max` (default 100.0):

$$w_\text{coll}^{k+1} = \min\!\left(w_\text{coll}^k \cdot \text{penalty\_scale},\; w_\text{coll}^\text{max}\right)$$

With defaults this produces the sequence: 1 → 3 → 9 → 27 → 81 → 100 → ... over 10 outer iterations.

Early outer iterations (low $w_\text{coll}$) allow the trajectory to find a smooth, geometrically reasonable path while tolerating some collision overlap. Later iterations (high $w_\text{coll}$) push all remaining penetrations to zero. This avoids local minima that would arise from starting with a high penalty.

---

## 7. Dimensionality Reduction via Smooth-Min

A robot collision model may produce hundreds of pairwise distances (e.g. 252 sphere-sphere pairs). Computing the Jacobian of all $P$ distances — shape $[P, n]$ — is expensive in both memory and compile time.

Instead, distances are aggregated per **collision group** using a **smooth minimum**:

$$\text{smooth\_min}(d) = -\tau \cdot \log\!\sum_i \exp\!\!\left(\frac{-d_i}{\tau}\right) = -\tau \cdot \text{logsumexp}\!\left(\frac{-d}{\tau}\right)$$

where $\tau$ is `smooth_min_temperature` (default 0.05). As $\tau \to 0$ this approaches the true minimum; larger $\tau$ gives smoother gradients.

**Groups** are:
- One group for all self-collision pairs
- One group per world geometry type (e.g. boxes, spheres)

This reduces the Jacobian shape from $[P, n]$ (e.g. $[252, 7]$) to $[G, n]$ (e.g. $[3, 7]$), a **50–100x reduction** in Jacobian memory and compile time.

The smooth-min is differentiable everywhere and provides a conservative approximation: it always returns a value $\leq$ the true minimum, so the optimizer is never over-optimistic about safety margins.

---

## 8. Batched Optimization and Initialization

### 8.1 Batch Parallelism

`sco_trajopt` accepts a batch of $B$ initial trajectories (`init_trajs: [B, T, DOF]`) and optimizes all of them in parallel via `jax.vmap` over the inner L-BFGS solve. Diverse initializations improve the chance of finding a globally good (low-cost, collision-free) solution.

The best trajectory is selected by evaluating the full nonlinear cost on all $B$ final trajectories and returning the one with the lowest cost.

### 8.2 Linear Interpolation Initialization

`make_init_trajs` creates a batch of $B$ linearly-interpolated trajectories between `start` and `goal`, with independent Gaussian noise added to each:

```python
base = start * (1 - t) + goal * t        # straight-line in joint space
trajs = base + normal(key, shape) * noise_scale
```

### 8.3 Cartesian IK Initialization (bench_trajopt)

The benchmark uses a more sophisticated initialization that produces collision-aware waypoints:

1. **FK** at start and goal to get end-effector SE(3) poses.
2. **Cartesian interpolation** — position lerp + SO(3) SLERP (via log/exp on the Lie algebra).
3. **Batched MPPI IK** — solves all $T$ IK subproblems in one GPU call, warm-started from linear joint-space interpolation, with self + world collision as a soft constraint.
4. **Tiling + noise** — the single IK-solved trajectory is tiled to $[B, T, n]$ and perturbed.

---

## 9. JAX Implementation Details

### Static vs. Traced Values

`TrajOptConfig` is a frozen dataclass passed as a **static argument** to `jax.jit`. Any change to the config triggers recompilation. This allows the compiler to unroll inner loops (L-BFGS history size `m_lbfgs`, inner iterations `n_inner_iters`) and specialize the computation graph completely.

### `jax.lax.scan` for the Outer Loop

The outer SCO loop uses `jax.lax.scan` with the trajectory batch and current collision weight as carry state. This avoids Python-level iteration at JIT time and compiles the entire outer loop into a single fused computation.

### `jax.lax.scan` for the Inner L-BFGS Loop

The inner L-BFGS loop also uses `jax.lax.scan`, with the L-BFGS buffers, current iterate, and best-iterate tracker as carry. The fixed-length scan is essential for JIT compatibility.

### Endpoint Re-pinning

After each outer iteration, endpoints are re-pinned:

```python
new_trajs = new_trajs.at[:, 0, :].set(start).at[:, -1, :].set(goal)
```

This corrects any tiny floating-point drift that accumulates when the gradient mask isn't perfectly exact.

### `jax.vmap` Layout

```
outer_step:
    _compute_coll_dists_and_jacs: vmap over B, vmap over T
    _lbfgs_inner_solve:           vmap over B
        line search:              vmap over 5 alpha candidates
```

---

## 10. Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `n_outer_iters` | 10 | Number of linearize-and-solve outer iterations |
| `n_inner_iters` | 30 | L-BFGS steps per outer iteration |
| `m_lbfgs` | 6 | L-BFGS history size (curvature pairs) |
| `w_smooth` | 1.0 | Overall smoothness weight |
| `w_acc` | 0.5 | Relative weight of acceleration in smoothness |
| `w_jerk` | 0.1 | Relative weight of jerk in smoothness |
| `w_collision` | 1.0 | Initial collision penalty weight |
| `w_collision_max` | 100.0 | Maximum collision penalty weight |
| `penalty_scale` | 3.0 | Per-outer-iteration collision weight multiplier |
| `collision_margin` | 0.01 | Safety buffer for collision activation (metres) |
| `w_trust` | 0.5 | Trust-region penalty weight |
| `w_limits` | 1.0 | Joint-limit violation penalty weight |
| `smooth_min_temperature` | 0.05 | Smooth-min temperature for per-group aggregation |

### Tuning Tips

- **Increase `n_outer_iters`** if trajectories still penetrate obstacles at convergence — the linearization needs more refinement steps.
- **Increase `w_collision` / `penalty_scale`** to push harder on collision avoidance from the start; lower these if the solver gets stuck in poor local minima.
- **Increase `n_inner_iters`** if the inner subproblem is not being solved to near-optimality (watch the cost drop per outer iteration).
- **Lower `smooth_min_temperature`** for tighter collision margins; raise it if gradients become noisy near obstacles.
- **Increase `m_lbfgs`** for better curvature approximation at the cost of higher memory and compile time.
