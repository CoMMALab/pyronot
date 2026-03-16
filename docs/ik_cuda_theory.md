# IK Solver Theory and CUDA Reimplementation

This document explains the mathematical theory behind each inverse kinematics solver in pyronot and describes the algorithmic modifications made when porting from JAX (CPU/GPU via XLA) to hand-written CUDA kernels.

---

## Table of Contents

1. [Common Foundations](#1-common-foundations)
2. [LS-IK: Least-Squares / Levenberg-Marquardt](#2-ls-ik-levenberg-marquardt)
3. [SQP-IK: Sequential Quadratic Programming](#3-sqp-ik-sequential-quadratic-programming)
4. [HJCD-IK: Hamiltonian Jacobian Coordinate Descent](#4-hjcd-ik-hamiltonian-jacobian-coordinate-descent)
5. [Multi End-Effector Extension](#5-multi-end-effector-extension)
6. [CUDA Architecture and Memory Model](#6-cuda-architecture-and-memory-model)
7. [Summary of JAX → CUDA Algorithm Changes](#7-summary-of-jax--cuda-algorithm-changes)

---

## 1. Common Foundations

### 1.1 SE(3) Residual via Lie Algebra Log-Map

All solvers share the same task-space residual. Given the current end-effector transform $T_\text{actual} \in SE(3)$ and the target $T_\text{target} \in SE(3)$, the 6-vector residual is:

$$f = \log\!\left(T_\text{actual}^{-1} \cdot T_\text{target}\right) \in \mathfrak{se}(3)$$

The log-map produces a 6-vector whose first three components are the translational error and last three are the rotation error in the Lie algebra $\mathfrak{so}(3)$. This is singularity-free (unlike Euler angles) and handles all orientations uniformly.

### 1.2 Jacobian

The geometric Jacobian $J \in \mathbb{R}^{6 \times n_\text{act}}$ maps joint velocities to end-effector velocities in the task frame. Only joints that are ancestors of the target link contribute non-zero columns.

- **JAX**: Computed via reverse-mode autodiff (`jax.vjp`) with 6 backward passes — one per row of $f$.
- **CUDA**: Hand-coded using exponential coordinates: each revolute joint $i$ contributes a column $[z_i \times (p_\text{ee} - p_i),\ z_i]^T$ where $z_i$ is the joint axis and $p_i$ is the joint origin in world frame.

### 1.3 Adaptive Position / Orientation Weighting

The residual components have different units (metres vs. radians) and different convergence rates. A diagonal weight matrix $W = \text{diag}(w_p, w_p, w_p, w_r, w_r, w_r)$ is applied to balance them:

$$w_r = \text{clip}\!\left(\frac{\|f_\text{pos}\|}{\|f_\text{ori}\|}, 0.05, 1\right)$$

This up-weights orientation when position is near-converged and vice versa. In practice it prevents orientation from dominating early in the search when position errors are still large.

### 1.4 Jacobi Column Scaling

To handle the large difference in joint sensitivity between (e.g.) a prismatic joint near the base and a revolute joint at the wrist, all methods scale the Jacobian columns before solving:

$$s_a = \|J_{:,a}\|_2 + \varepsilon, \qquad \tilde{J} = J \cdot \text{diag}(s)^{-1}$$

The normal equations are solved in the scaled space and the step is un-scaled:

$$\tilde{p} = (\tilde{J}^T \tilde{J} + \lambda I)^{-1}(-\tilde{J}^T f_w), \qquad p = \tilde{p} \odot s^{-1}$$

### 1.5 Trust-Region Clipping

After computing a Newton-like step $p$, it is clipped component-wise to a radius $R$ that adapts to the current error magnitude:

| Condition | $R$ |
|---|---|
| $\|f_p\| > 10\,\text{mm}$ or $\|f_r\| > 0.6\,\text{rad}$ | 0.38 |
| $\|f_p\| > 1\,\text{mm}$ or $\|f_r\| > 0.25\,\text{rad}$ | 0.22 |
| $\|f_p\| > 0.2\,\text{mm}$ or $\|f_r\| > 0.08\,\text{rad}$ | 0.12 |
| otherwise | 0.05 |

Large steps are allowed far from the solution to escape flat regions; small steps near the solution for precision.

### 1.6 Multi-Seed Strategy

All solvers run on a *population* of initial configurations (seeds) in parallel. Seeds come from three sources:

- **Warm seeds** (tight, $\sigma = 0.05$): perturbations of the previous solution — for continuity in trajectory following.
- **Warm seeds** (loose, $\sigma = 0.3$): wider perturbations — escape local minima near the last solution.
- **Random seeds**: uniform samples from joint limits — global coverage.

The best solution across all seeds is returned.

---

## 2. LS-IK: Levenberg-Marquardt

### 2.1 Algorithm (JAX)

LS-IK is a standard Gauss-Newton solver with Levenberg-Marquardt damping. There is no coarse phase — all seeds go directly into refinement.

**Each outer iteration:**

1. Compute weighted residual $f_w = W f$ and weighted Jacobian $\tilde{J}_w = W \tilde{J}$.
2. Form the damped normal equations:
   $$H \tilde{p} = -g, \qquad H = \tilde{J}_w^T \tilde{J}_w + \lambda I, \quad g = \tilde{J}_w^T f_w$$
3. Solve for $\tilde{p}$ via Cholesky (or LU for small $n_\text{act}$).
4. Un-scale: $p = \tilde{p} \odot s^{-1}$, then clip to trust-region radius $R$.
5. **Line search**: evaluate $q + \alpha p$ for $\alpha \in \{1, 0.5, 0.25, 0.1, 0.025\}$ using JAX `vmap` (all five in parallel). Accept the largest $\alpha$ that reduces the error.
6. Update $\lambda$: halve on success, triple on failure. Bounds: $[10^{-10}, 10^6]$.

**Convergence**: $\|f_p\| < 1\,\text{mm}$ and $\|f_r\| < 0.05\,\text{rad}$ for all end-effectors.

### 2.2 CUDA Modifications

| Aspect | JAX | CUDA |
|---|---|---|
| Line search | 5-point `vmap` (parallel) | Sequential loop; exit on sufficient descent |
| Cholesky solve | `jnp.linalg.solve` (cuBLAS) | Hand-coded in-kernel Cholesky (float64) |
| Jacobian | Reverse-mode autodiff | Hand-coded FK + exponential-coordinate Jacobian |
| Normal equations | Float32 + XLA | Float64 accumulation to avoid ill-conditioning |
| Parallelism | `vmap` over seeds | One thread per seed; shared memory for robot model |

The line search becomes sequential in the kernel because CUDA does not support dynamic parallelism within a warp for small sub-tasks like this. Since the five evaluations are ordered by preference, early exit on the first improvement keeps the cost low in practice.

---

## 3. SQP-IK: Sequential Quadratic Programming

### 3.1 Algorithm (JAX)

SQP-IK extends LS-IK to enforce joint limits as **hard box constraints** on each step rather than clamping the configuration post-hoc. Joint limits are written as:

$$q + p \in [\mathbf{q}_\text{lower},\ \mathbf{q}_\text{upper}] \quad \Longleftrightarrow \quad p \in [\mathbf{q}_\text{lower} - q,\ \mathbf{q}_\text{upper} - q]$$

At each outer iteration the same damped normal equations are formed, but the step $p$ is then refined by a small inner QP loop:

**Inner QP iterations** (typically 2–3):

1. Solve unconstrained step: $p = H^{-1}(-g)$ (same as LS-IK, in scaled space).
2. Clamp to box: $p \leftarrow \text{clip}(p,\, \ell,\, u)$ where $\ell, u$ are the scaled joint-limit bounds.
3. **Active-set refinement**: identify *active* joints (those that hit a bound). Fix their contribution. Re-solve the reduced system for the *free* joints only.
4. Repeat from step 2 with updated active set.

Because the step always stays inside the joint limits, there is no need for a post-hoc projection that could corrupt the LM update. This leads to faster convergence when the solution is near a joint limit.

The safe step size for the inner loop is $\alpha = 1/(n_\text{active} + \lambda)$, which guarantees descent for the constrained sub-problem.

### 3.2 CUDA Modifications

The SQP kernel is structurally identical to the LS-IK kernel with an added inner loop after the unconstrained solve:

| Aspect | JAX | CUDA |
|---|---|---|
| Active-set solve | Masked matrix solve via `jnp.where` | Explicit reduced-system solve; fixed-size buffers |
| Bound representation | Scaled inside JAX ops | Precomputed `lb_s`, `ub_s` per joint in scaled space |
| Inner iterations | Unrolled via `jax.lax.fori_loop` | Plain `for` loop in kernel |

All other modifications (line search, Cholesky, float64 normal equations) are the same as LS-IK CUDA.

---

## 4. HJCD-IK: Hamiltonian Jacobian Coordinate Descent

HJCD-IK is the most sophisticated solver. It uses a two-phase strategy: a fast **coarse phase** (coordinate descent) to get close to the solution, followed by a precise **refinement phase** (Levenberg-Marquardt with stall recovery).

### 4.1 Phase 1: Hamiltonian Coordinate Descent (Coarse)

Rather than computing a full Newton step in joint space, the coarse phase selects and updates **one joint per iteration** — the joint most aligned with the task-space gradient.

**Scoring**: For each active joint $a$:

$$\text{score}_a = \frac{|J_w^T f_w|_a}{\|J_{w,: a}\|}$$

This is the normalised gradient component: the numerator is how much that joint reduces the error, the denominator is its sensitivity (prevents preferring high-gain joints that cause oscillation).

**Momentum update**: The selected joint $a^*$ gets a momentum-based step:

$$p_{a^*} \leftarrow 0.9\, p_{a^*} - 0.35\, \frac{g_{a^*}}{\|J_{w,: a^*}\|^2}$$
$$q_{a^*} \leftarrow \text{clip}(q_{a^*} + p_{a^*},\, q^\text{lower}_{a^*},\, q^\text{upper}_{a^*})$$

The coefficient 0.9 provides Nesterov-like momentum; 0.35 is a conservative gradient gain that keeps the update stable. Momentum is per-joint and tracked across iterations within a seed.

**Early exit**: The loop terminates early when $\|f_p\| < 20\,\text{mm}$ **and** $\|f_r\| < \pi/2\,\text{rad}$ for all end-effectors — a coarse convergence threshold that hands off to the LM refinement.

**Why coordinate descent?** The coarse search explores the configuration manifold quickly without solving a full linear system per step. It handles non-convexity better than gradient descent while being cheaper than Newton. The momentum term helps cross saddle points.

### 4.2 Phase 2: Levenberg-Marquardt with Stall Recovery (Refinement)

This phase runs standard LM (same as LS-IK) with three additions:

**Per-EE adaptive weighting**: Each end-effector maintains its own weight $w_\text{sg}^{(i)}$. When end-effector $i$ has converged in position but not orientation, its orientation weight is boosted independently, without affecting other end-effectors.

**Soft joint-limit prior**: A diagonal regularizer is added to the normal equations:

$$H \leftarrow H + w_\text{prior} \cdot \text{diag}\!\left(\left(\frac{q - q_\text{mid}}{q_\text{half\_range}}\right)^2\right)$$

This gently biases the solution toward the centre of each joint's range. The prior weight is small enough to be dominated by the task objective but prevents the solver from wandering to joint limits when the task is under-constrained.

**Stall detection and random kicks**: After 6 consecutive iterations with no improvement:

1. A Gaussian perturbation is sampled: $q \leftarrow q + \mathcal{N}(0,\, \sigma_\text{kick}^2 I)$.
2. $\lambda$ is reset to its initial value.
3. The *best-seen* configuration is tracked separately, so kicks cannot degrade the returned result.

Kicks allow the solver to escape flat regions and narrow valleys that LM alone cannot exit.

**Top-K selection**: Before refinement, the coarse solutions are ranked and only the top-K are kept. These are then replicated with small perturbations to fill the seed budget, ensuring the refinement phase explores near the best coarse solutions.

### 4.3 CUDA Modifications

HJCD uses **two separate CUDA kernels** — one per phase — because the two phases have very different compute profiles and termination conditions.

**Coarse kernel** (`hjcd_ik_coarse_kernel`):

| Aspect | JAX | CUDA |
|---|---|---|
| Momentum state | JAX array updated in `lax.fori_loop` | Per-thread `mom[MAX_ACT]` array |
| Per-EE ori gating | Computed lazily inside vmap | Explicit per-EE convergence flag checked each iteration |
| Joint selection | `jnp.argmax` | Linear scan over active joints (small $n_\text{act}$) |
| Early exit | `lax.cond` (XLA-friendly) | `break` on convergence flag |

**Refinement kernel** (`hjcd_ik_lm_cuda`):

| Aspect | JAX | CUDA |
|---|---|---|
| Kicks | Pre-sampled noise array passed in | Pre-generated Gaussian array passed as kernel argument; thread indexes into it |
| Best-config tracking | `jax.lax.cond` | Explicit `if (new_err < best_err)` with register copy |
| Stall counter | Python-level tracked via carry | Per-thread integer counter |
| Limit prior | Added inside `jnp.linalg.solve` call | Added to float64 diagonal before Cholesky |

The noise array for kicks is generated on the Python side before the kernel launch (using `jax.random`) and passed as a read-only argument. This avoids the need for a CUDA random state inside the kernel, which would require cuRAND device API and complicate the code significantly.

---

## 5. Multi End-Effector Extension

### 5.1 API

The solver accepts a tuple of target link indices `target_link_indices = (i_0, i_1, ..., i_{N-1})` and a corresponding tuple of target poses. All end-effectors must converge for the solution to be accepted.

### 5.2 Residual Stacking

For $N$ end-effectors the residual vector is stacked:

$$f = \begin{bmatrix} W_0 f_0 \\ W_1 f_1 \\ \vdots \\ W_{N-1} f_{N-1} \end{bmatrix} \in \mathbb{R}^{6N}$$

The Jacobian is correspondingly stacked row-wise. The normal equations $J^T J$ sum contributions from all end-effectors, so joints that affect multiple end-effectors receive stronger gradient signal.

### 5.3 Ancestor Masks

Each end-effector only has a partial kinematic chain. A per-EE ancestor mask $M_{i,j} \in \{0, 1\}$ is precomputed at the Python level:

$$M_{i,j} = 1 \iff \text{joint } j \text{ is an ancestor of EE } i$$

In the CUDA kernels, this mask is used to skip FK updates and zero out Jacobian columns for joints that don't affect a given EE. This reduces both compute and the effective system size in the normal equations.

### 5.4 Trust Region Under Multiple EEs

For LS-IK and SQP-IK, the trust-region radius is determined by the **maximum** position/orientation error across all EEs:

$$R = R\!\left(\max_i \|f_{p,i}\|,\ \max_i \|f_{r,i}\|\right)$$

This keeps the trust region open as long as any EE still needs large steps, preventing premature step-size reduction.

For HJCD refinement, per-EE weights $w_\text{sg}^{(i)}$ are maintained independently, so a converged EE does not suppress orientation updates for a still-unconverged EE.

---

## 6. CUDA Architecture and Memory Model

### 6.1 Grid Layout

```
Grid:  ( ceil(n_seeds / BLOCK_SIZE),  n_problems )
Block: ( BLOCK_SIZE,  1 )
```

Each thread handles exactly one seed for one IK problem. Multiple problems (e.g., a batch of target poses) are handled in the second grid dimension, with each block of threads sharing robot model data for one problem.

### 6.2 Shared Memory

Each block loads the robot model parameters cooperatively into shared memory at kernel launch:

- Joint twist parameters (axis, position, type)
- Parent transforms (fixed transforms between joints)
- Joint parent indices (kinematic tree structure)
- Target link indices per EE
- Ancestor masks per EE

Loading is done once per block (not once per thread). Threads in a block synchronise with `__syncthreads()` before proceeding to per-thread computation.

### 6.3 Thread-Private State

Each thread maintains in registers / local memory:

| Variable | Size | Purpose |
|---|---|---|
| `cfg[MAX_ACT]` | $n_\text{act}$ floats | Current joint configuration |
| `best_cfg[MAX_ACT]` | $n_\text{act}$ floats | Best seen configuration |
| `T_world[MAX_JOINTS × 7]` | 7-float quaternion+translation per joint | FK world transforms |
| `r[6 × MAX_EE]` | $6 N_\text{ee}$ floats | Stacked residuals |
| `J[6 × MAX_EE × MAX_ACT]` | $6 N_\text{ee} n_\text{act}$ floats | Stacked Jacobian |
| `H[MAX_ACT × MAX_ACT]` | $n_\text{act}^2$ floats (float64) | Normal equation matrix |

`MAX_ACT`, `MAX_JOINTS`, `MAX_EE` are compile-time constants that set an upper bound on robot complexity. Actual sizes are passed as kernel arguments and loop bounds are set accordingly.

### 6.4 Float64 Normal Equations

The Jacobian and residuals are computed in float32 (matching the FK precision). However, the normal equation matrix $H = J^T J + \lambda I$ is accumulated in float64. This is critical because:

- $J^T J$ sums $6 N_\text{ee}$ outer products; rounding errors accumulate.
- The condition number of $H$ can be large when $\lambda$ is small.
- float32 can lose the $\lambda I$ term entirely when diagonal entries of $J^T J$ are large.

Only the Cholesky factorisation and back-substitution use float64; results are cast back to float32 before the configuration update.

---

## 7. Summary of JAX → CUDA Algorithm Changes

| Feature | JAX (CPU/XLA-GPU) | CUDA Kernel | Reason for Change |
|---|---|---|---|
| Multi-seed parallelism | `jax.vmap` over seeds | One thread per seed | Direct GPU thread mapping; avoids XLA overhead |
| Line search | 5 candidates evaluated in parallel via `vmap` | Sequential loop with early exit | No nested parallelism in kernel; acceptable cost |
| Cholesky solver | `jnp.linalg.solve` → cuBLAS | Hand-coded in-kernel (float64) | Avoids kernel launch overhead for tiny systems ($n_\text{act} \leq 12$) |
| Jacobian computation | Reverse-mode autodiff (6 backward passes) | Hand-coded exponential-coordinate formula | Full control over memory layout; ~3× faster for small $n$ |
| Random kicks (HJCD) | JAX PRNG inside `lax.while_loop` | Pre-generated noise array passed as argument | Avoids cuRAND device state in kernel |
| Line search for HJCD | `vmap` over 5 alphas | Sequential | Same as LS-IK |
| Per-EE weight updates | Recomputed each iteration inside vmapped scan | Explicit per-thread per-EE flags | Avoids redundant float ops in hot loop |
| Top-K selection | NumPy argsort on JAX output | Performed on Python side before refinement kernel | Simpler; selection is not on the critical path |
| Stall counter | Carried in `lax.scan` state | Per-thread integer register | Natural in imperative kernel |
| Robot model | JAX arrays (heap-allocated) | Shared memory per block | Dramatically reduces global memory bandwidth |
| Normal equation precision | float32 throughout | float32 for J, float64 for $H = J^T J$ | Stability when $\lambda$ small or columns poorly scaled |
