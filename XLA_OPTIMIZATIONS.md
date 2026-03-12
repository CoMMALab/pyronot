# XLA / CUDA Optimizations for HJCD-IK

Applied to the CUDA and JAX implementations in `src/pyronot/cuda_kernels/` and
`src/pyronot/optimization_engines/_hjcd_ik.py`.

---

## Summary Table

| # | File(s) | Change | Expected Impact |
|---|---------|--------|----------------|
| 1 | `_hjcd_ik.py:672` | On-device top-K via `jax.lax.top_k` | **High** — eliminates GPU→CPU sync |
| 2 | `_hjcd_ik_cuda_kernel.cu` | Shared memory for robot params — coarse kernel | **Medium-High** — saves global DRAM bandwidth per FK/Jacobian call |
| 3 | `_hjcd_ik_cuda_kernel.cu` | Shared memory for robot params — LM kernel | **Medium-High** — same, applied to inner LM loop |
| 4 | `_fk_cuda_kernel.cu` | Shared memory for robot params — FK kernel | **Medium** — batch FK benefits from L1 broadcast |
| 5 | `build_hjcd_ik_cuda.sh` | `--use_fast_math`, `-O3`, `-arch=native` | **Medium** — FMA + fast trig + native ISA |
| 6 | `build_fk_cuda.sh` | `--use_fast_math`, `-O3`, `-arch=native` | **Medium** — same for FK library |

---

## Detailed Changes

### 1. On-device Top-K Selection (`_hjcd_ik.py`)

**Before:**
```python
top_k_indices = jnp.argsort(coarse_errors)[:top_k]
```

**After:**
```python
_, top_k_indices = jax.lax.top_k(-coarse_errors, top_k)
```

`jnp.argsort` forces XLA to materialise the full sorted array, which requires
a device→host transfer to allocate the output, then a host→device transfer to
return the indices before Phase 2 can launch.  `jax.lax.top_k` maps to XLA's
`TopK` op and runs entirely on-device with no implicit synchronisation point.

**Why it matters:** Every IK call had a hidden GPU pipeline stall between
Phase 1 and Phase 2.  This is now a single fused on-device operation.

---

### 2–3. Shared Memory for Robot Parameters — IK Kernels (`_hjcd_ik_cuda_kernel.cu`)

Applied to both `hjcd_ik_coarse_kernel` and `hjcd_ik_lm_kernel`.

**Pattern:**
```cuda
__shared__ float s_twists       [MAX_JOINTS * 6];   // 768 B
__shared__ float s_parent_tf    [MAX_JOINTS * 7];   // 896 B
__shared__ int   s_parent_idx   [MAX_JOINTS];        // 128 B
__shared__ int   s_act_idx      [MAX_JOINTS];        // 128 B
__shared__ float s_mimic_mul    [MAX_JOINTS];        // 128 B
__shared__ float s_mimic_off    [MAX_JOINTS];        // 128 B
__shared__ int   s_mimic_act_idx[MAX_JOINTS];        // 128 B
__shared__ int   s_topo_inv     [MAX_JOINTS];        // 128 B
__shared__ int   s_ancestor_mask[MAX_JOINTS];        // 128 B
__shared__ float s_target_T[7];                      //  28 B
__shared__ float s_lower   [MAX_ACT];                //  64 B
__shared__ float s_upper   [MAX_ACT];                //  64 B
__shared__ int   s_fixed_mask[MAX_ACT];              //  64 B
// Total: ~2.8 KB/block (well within 48 KB limit)

// Cooperative load: all threads participate before early-exit guard.
for (int i = threadIdx.x; i < n_joints * 6; i += blockDim.x)
    s_twists[i] = twists[i];
// ... (same for all arrays)
__syncthreads();  // Barrier before any thread begins FK.

// Early-exit moved AFTER sync so out-of-range threads contribute to the load.
const int s = blockIdx.x * blockDim.x + threadIdx.x;
if (s >= n_seeds) return;
```

All calls to `compute_residual_and_jacobian`, `compute_residual_only`, and
per-joint limit clamps now receive shared-memory pointers instead of global
memory pointers.

**Why it matters:**
- `twists` and `parent_tf` are read on every call to `fk_single` (inside every
  Jacobian computation and every line-search FK evaluation).
- Coarse kernel: up to `k_max=20` iterations × 1 FK+Jacobian = 20 global reads
  per seed.  With 128 seeds/block, that is 2560 global reads of the same data —
  now served from L1.
- LM kernel: up to `max_iter=40` iterations × 6 FK calls (Jacobian + 5
  line-search) = 240 global reads per seed.  Now served from L1.

**Structural note:** The `if (s >= n_seeds) return;` guard was moved to *after*
`__syncthreads()`.  This is required for correctness: if the guard fired before
the sync, threads with `s >= n_seeds` would exit the block, leaving the
remaining threads waiting forever on `__syncthreads()`.

---

### 4. Shared Memory for Robot Parameters — FK Kernel (`_fk_cuda_kernel.cu`)

Same pattern as above but for the standalone FK kernel.  Added
`#define FK_MAX_JOINTS 64` (larger than the IK limit since FK has no quadratic
per-thread memory from normal equations).

**Footprint:** ~3.7 KB/block with 64 joints.  Well within the 48 KB limit.

**Why it matters:** Large FK batches (e.g., 65 536 elements, as tested in
`test_fk_cuda.py`) repeat the robot-parameter reads for every batch element.
With 256 threads per block and 256 batch elements per block, the cooperative
load amortises 256 global reads into 1 shared-memory load per block.

---

### 5–6. Build Flags (`build_hjcd_ik_cuda.sh`, `build_fk_cuda.sh`)

Three changes per script:

| Flag | Effect |
|------|--------|
| `-O2` → `-O3` | More aggressive loop/instruction optimisation |
| `--use_fast_math` | Enables FMA (fused multiply-add) and fast reciprocal/`sqrt`/trig — relevant for `se3_exp`, `norm3`, and all residual arithmetic |
| `${GPU_ARCH:--arch=native}` | Targets the actual installed GPU ISA.  `-arch=native` (CUDA 11.6+) emits PTX and SASS for the physical GPU, unlocking hardware-specific intrinsics (Tensor Cores, warp-level reductions, etc.).  Users on older CUDA can override: `GPU_ARCH=-arch=sm_80 bash build_hjcd_ik_cuda.sh` |

**Why `--use_fast_math` is safe here:** The IK solver does not require
IEEE-754 rounding guarantees.  The FK math (quaternion products, `se3_exp`) and
residual computation tolerate the ≈2 ULP error of fast-math functions.  The
only precision-sensitive path (normal equations + Cholesky) uses `double` and
is unaffected because `-use_fast_math` only affects single-precision ops.

---

## Rebuild Required

After pulling these changes you must recompile both shared libraries:

```bash
# From repo root
bash src/pyronot/cuda_kernels/build_fk_cuda.sh
bash src/pyronot/cuda_kernels/build_hjcd_ik_cuda.sh
```

If your CUDA version is older than 11.6 and does not support `-arch=native`,
pass the appropriate architecture manually:

```bash
GPU_ARCH=-arch=sm_80 bash src/pyronot/cuda_kernels/build_fk_cuda.sh
GPU_ARCH=-arch=sm_80 bash src/pyronot/cuda_kernels/build_hjcd_ik_cuda.sh
```

---

## What Was Not Changed

The following optimisations from the initial analysis were considered but
deferred:

- **Warp-level reductions for column norms** — only 6 scalar ops per column,
  overhead of warp-shuffle setup likely exceeds savings for `n_act ≤ 16`.
- **Buffer aliasing via `ResultBuffer`** — requires XLA FFI API changes that
  could break existing JAX dispatch caching; benefit is marginal (one allocation
  per call).
- **Adaptive precision in LM** — the current float64 Cholesky is already fast
  relative to FK; switching to float32 early in convergence adds code complexity
  for modest gain.
