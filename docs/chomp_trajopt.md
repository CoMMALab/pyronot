# CHOMP TrajOpt: Theory and Implementation Notes

This document explains the CHOMP trajectory optimizer used in pyroffi, with emphasis on how the two backends differ:

- JAX implementation in `src/pyroffi/optimization_engines/_chomp_optimization.py`
- CUDA implementation in `src/pyroffi/cuda_kernels/_chomp_trajopt_cuda_kernel.cu`

The goal is to clarify the optimization objective, update rule, continuation strategy, and the main computational trade-offs.

**Reference:** Ratliff et al., "CHOMP: Gradient Optimization Techniques for Efficient Motion Planning".

---

## Table of Contents

1. [Problem Setup](#1-problem-setup)
2. [CHOMP Objective in This Codebase](#2-chomp-objective-in-this-codebase)
3. [Covariant Update Rule](#3-covariant-update-rule)
4. [Collision Model and Hinge Penalty](#4-collision-model-and-hinge-penalty)
5. [Per-Iteration CHOMP Loop](#5-per-iteration-chomp-loop)
6. [JAX Backend: Theoretical Behavior](#6-jax-backend-theoretical-behavior)
7. [CUDA Backend: Theoretical Behavior](#7-cuda-backend-theoretical-behavior)
8. [Why CUDA Can Be Slower Than JAX Here](#8-why-cuda-can-be-slower-than-jax-here)
9. [Configuration Parameters and Their Effects](#9-configuration-parameters-and-their-effects)
10. [Practical Guidance](#10-practical-guidance)

---

## 1. Problem Setup

We optimize a trajectory

$$
\mathbf{q} = [q_0, q_1, \dots, q_{T-1}] \in \mathbb{R}^{T \times n}
$$

where:

- $T$ is the number of waypoints
- $n$ is the number of actuated DOFs
- endpoints are fixed: $q_0 = q_{\text{start}}$, $q_{T-1} = q_{\text{goal}}$

CHOMP solves a smooth unconstrained optimization with soft penalties for limits and collisions.

---

## 2. CHOMP Objective in This Codebase

The objective used by both backends is:

$$
J(\mathbf{q}) =
 w_{\text{smooth}} J_{\text{smooth}}(\mathbf{q})
 + w_{\text{limits}} J_{\text{limits}}(\mathbf{q})
 + w_{\text{coll}} J_{\text{coll}}(\mathbf{q}).
$$

### 2.1 Smoothness term

The smoothness part penalizes acceleration and jerk with a 5-point stencil:

$$
a_t = \frac{-q_t + 16q_{t+1} - 30q_{t+2} + 16q_{t+3} - q_{t+4}}{12},
$$

$$
J_{\text{smooth}} =
 w_{\text{acc}} \sum_t \|a_t\|^2
 +
 w_{\text{jerk}} \sum_t \|a_{t+1} - a_t\|^2.
$$

This makes trajectories smooth in higher-order derivatives, not just velocity.

### 2.2 Joint-limit term

Soft squared exceedance:

$$
J_{\text{limits}} =
\sum_{t,i}
\left[
\max(0, q_{t,i} - u_i) + \max(0, \ell_i - q_{t,i})
\right]^2.
$$

### 2.3 Collision term

For signed distance $d$ and margin $m$, a quadratic hinge is used:

$$
\phi(d) = \max(0, m - d)^2,
\quad
J_{\text{coll}} = \sum_{t,p} \phi\big(d_{t,p}(q_t)\big).
$$

This starts penalizing before penetration (inside the margin).

---

## 3. Covariant Update Rule

Classic CHOMP applies a metric-preconditioned gradient step:

$$
\mathbf{q}_{k+1} = \mathbf{q}_k - \alpha M^{-1} \nabla J(\mathbf{q}_k).
$$

In this codebase, $M$ is a trajectory smoothness precision matrix over interior timesteps. Intuitively:

- raw gradients can be noisy waypoint-wise
- multiplying by $M^{-1}$ produces smoother, geometry-aware updates

When `use_covariant_update=True`:

- JAX backend solves linear systems with `jnp.linalg.solve`
- CUDA backend uses a conjugate-gradient (CG) solve per DOF on the interior timeline

When disabled, update direction is plain steepest descent: $-\nabla J$.

---

## 4. Collision Model and Hinge Penalty

The robot is represented with spherized link geometry (`RobotCollisionSpherized`). Distances are computed for:

- self-collision active link pairs
- world obstacles (sphere, capsule, box, half-space)

Per collision primitive pair, signed distance is evaluated; then the margin hinge is applied.

The continuation schedule gradually increases collision weight:

$$
w_{\text{coll}} \leftarrow \min(\gamma w_{\text{coll}}, w_{\text{coll,max}}),
$$

where $\gamma = \text{collision_penalty_scale}$.

This helps avoid poor local minima early in optimization.

---

## 5. Per-Iteration CHOMP Loop

Each CHOMP iteration in both backends follows this pattern:

1. Evaluate current cost and gradient.
2. Clip gradient norm (optional).
3. Build descent direction (covariant or Euclidean).
4. Evaluate several step sizes (line search candidates).
5. Accept best improving step.
6. Re-pin endpoints exactly.
7. Update collision continuation weight.
8. Optionally early-stop on repeated non-improvement.

This is a first-order method with structured preconditioning.

---

## 6. JAX Backend: Theoretical Behavior

The JAX path computes gradients by automatic differentiation through the full objective:

- exact reverse/forward AD through smoothness, limits, FK, and collision distance graph
- line-search candidates are batched by `vmap`
- optimization loops use `lax.scan`/`jit`

The important theoretical property is gradient quality:

$$
\nabla J_{\text{coll}} \text{ is obtained by AD of the full composite map } q \to d(q) \to \phi(d).
$$

No finite-difference truncation error is introduced.

---

## 7. CUDA Backend: Theoretical Behavior

The CUDA path follows the same objective structure, but not the same differentiation mechanism.

### 7.1 Smoothness and limits gradient

Analytic expressions are used directly in-kernel for smoothness and limit gradients.

### 7.2 Collision gradient

Collision gradient is approximated by finite differences per timestep and DOF around the current waypoint configuration:

$$
\frac{\partial J_{\text{coll}}}{\partial q_i}
\approx
\frac{J_{\text{coll}}(q + \epsilon e_i) - J_{\text{coll}}(q)}{\epsilon}
\quad
\text{(one-sided in the current kernel)}.
$$

This requires repeated FK + collision evaluation for each perturbed coordinate.

### 7.3 Line search

Several candidate step sizes are evaluated (parallel across a few threads), each requiring full trajectory cost evaluation.

So CUDA currently trades exact AD for repeated numerical evaluations.

---

## 8. Why CUDA Can Be Slower Than JAX Here

Even with GPU execution, the current CUDA CHOMP can be slower because the operation count is dominated by repeated collision/FK work.

At a high level, per iteration:

- base collision evaluations across all timesteps
- plus finite-difference perturbations for each interior DOF-time coordinate
- plus multiple full-cost line-search candidate evaluations

This scales roughly like:

$$
\mathcal{O}\big(T \cdot (1 + n) \cdot C_{\text{coll}}\big)
+ \mathcal{O}\big(K \cdot T \cdot C_{\text{coll}}\big),
$$

where:

- $n$ = DOF
- $T$ = timesteps
- $K$ = line-search candidates
- $C_{\text{coll}}$ = cost of one FK + collision pass

By contrast, JAX AD can reuse intermediate structure and often emits highly optimized fused kernels for the derivative pipeline.

In short: the CUDA backend is compute-heavy due to numeric gradient estimation, not because CHOMP theory is different.

---

## 9. Configuration Parameters and Their Effects

Main CHOMP parameters (`ChompTrajOptConfig`):

- `n_iters`: optimization iterations
- `step_size`: base step before candidate scaling
- `w_smooth`, `w_acc`, `w_jerk`: trajectory smoothness shaping
- `w_limits`: joint-limit softness
- `w_collision`, `w_collision_max`, `collision_penalty_scale`: continuation schedule
- `collision_margin`: safety activation distance
- `use_covariant_update`, `smoothness_reg`: preconditioned update behavior
- `grad_clip_norm`: robustness against extreme gradients
- `max_delta_per_step`: trust-region style per-iteration motion bound
- `early_stop_patience`, `min_cost_improve`: convergence/termination behavior

Tuning note:

- larger `w_collision_max` and margin usually improve clearance but can increase roughness or runtime
- smaller `step_size` can stabilize but may need more iterations
- disabling covariant update often hurts smoothness and convergence rate

---

## 10. Practical Guidance

For theoretical parity between backends, both should optimize the same objective and continuation schedule (which they do conceptually). The remaining difference is mainly gradient computation strategy:

- JAX: AD-based collision gradients
- CUDA: finite-difference collision gradients (current)

The most impactful path to close speed/quality gaps is replacing finite-difference collision gradients in CUDA with analytic or autodiff-equivalent gradient propagation through FK and distance primitives.

That change preserves CHOMP theory while reducing repeated collision evaluations substantially.
