/**
 * Stein Variational Gradient Descent (SVGD) IK CUDA kernel with Jacobian guidance.
 *
 * Implements Jacobian-guided SVGD for region-based inverse kinematics.
 * The algorithm transports particles to cover the kinematic constraint manifold
 * uniformly using:
 *   - Gradient of log-target (task residuals)
 *   - RBF kernel for particle repulsion
 *
 * Correct SVGD update:
 *   phi(x_i) = (1/N) sum_j [ k(x_j, x_i) * grad_logp(x_j)
 *                            + grad_{x_j} k(x_j, x_i) ]
 * where:
 *   log p(q) = -||r(q)||^2
 *   grad log p(q) = -2 * J^T * r
 *   grad_{x_j} k(x_j, x_i) = k(x_j, x_i) * (x_i - x_j) / h^2
 *
 * Particles live in shared memory so all threads see the same evolving state.
 *
 * Multi-EE support: stacked residuals and Jacobians for all EEs simultaneously.
 *
 * Adaptive bandwidth: at each iteration the RBF bandwidth is set via the median
 * heuristic over current pairwise distances:
 *   h = median(||x_i - x_j||^2) / log(N),   bandwidth = sqrt(h)
 * Each thread sorts its own row of pairwise distances to find the per-particle
 * median; thread 0 then picks the median of those N values.  If the computed
 * bandwidth is degenerate (particles collapsed) the caller-supplied fallback
 * value is used instead.
 *
 * Build with:  bash src/pyronot/cuda_kernels/build_svgd_region_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"

#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits (same as other IK kernels)
// ---------------------------------------------------------------------------

#ifndef MAX_PARTICLES
#define MAX_PARTICLES 32
#endif

#ifndef MAX_LBFGS_M
#define MAX_LBFGS_M 8
#endif

// ---------------------------------------------------------------------------
// CUDA kernel: one thread per particle, one block per problem
// ---------------------------------------------------------------------------

__global__
void svgd_region_ik_kernel(
    const float* __restrict__ seeds,         // (n_problems, n_particles, n_act)
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ target_jnts,
    const int*   __restrict__ ancestor_masks,
    const float* __restrict__ target_Ts,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    int64_t n_iters,
    float bandwidth,
    float step_size,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    float*       __restrict__ out_ee,     // (n_problems * n_particles, 3)
    float*       __restrict__ out_target, // (n_problems * n_particles, 3)
    int n_problems, int n_particles, int n_joints, int n_act, int n_ee)
{
    // ---- Shared memory: robot parameters (loaded once per block) ----
    __shared__ float s_twists       [MAX_JOINTS * 6];
    __shared__ float s_parent_tf    [MAX_JOINTS * 7];
    __shared__ int   s_parent_idx   [MAX_JOINTS];
    __shared__ int   s_act_idx      [MAX_JOINTS];
    __shared__ float s_mimic_mul    [MAX_JOINTS];
    __shared__ float s_mimic_off    [MAX_JOINTS];
    __shared__ int   s_mimic_act_idx[MAX_JOINTS];
    __shared__ int   s_topo_inv     [MAX_JOINTS];
    __shared__ float s_target_Ts   [MAX_EE * 7];
    __shared__ int   s_target_jnts [MAX_EE];
    __shared__ int   s_ancestor_masks[MAX_EE * MAX_JOINTS];
    __shared__ float s_lower   [MAX_ACT];
    __shared__ float s_upper   [MAX_ACT];
    __shared__ int   s_fixed_mask[MAX_ACT];

    // ---- Shared memory: particle state, gradients, and adaptive bandwidth ----
    __shared__ float s_particles[MAX_PARTICLES * MAX_ACT];
    __shared__ float s_grad_logp[MAX_PARTICLES * MAX_ACT];
    // Per-particle median pairwise distance² (used for bandwidth estimation)
    __shared__ float s_row_med[MAX_PARTICLES];
    // Adaptive bandwidth (written by thread 0, read by all)
    __shared__ float s_bandwidth;

    // Cooperative load of robot parameters
    for (int i = threadIdx.x; i < n_joints * 6; i += blockDim.x) s_twists[i]    = twists[i];
    for (int i = threadIdx.x; i < n_joints * 7; i += blockDim.x) s_parent_tf[i] = parent_tf[i];
    for (int i = threadIdx.x; i < n_joints;     i += blockDim.x) {
        s_parent_idx[i]    = parent_idx[i];
        s_act_idx[i]       = act_idx[i];
        s_mimic_mul[i]     = mimic_mul[i];
        s_mimic_off[i]     = mimic_off[i];
        s_mimic_act_idx[i] = mimic_act_idx[i];
        s_topo_inv[i]      = topo_inv[i];
    }
    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[i];
        s_upper[i]      = upper[i];
        s_fixed_mask[i] = fixed_mask[i];
    }
    const int p = blockIdx.y;
    for (int i = threadIdx.x; i < n_ee * 7; i += blockDim.x)
        s_target_Ts[i] = target_Ts[p * n_ee * 7 + i];
    for (int i = threadIdx.x; i < n_ee; i += blockDim.x)
        s_target_jnts[i] = target_jnts[i];
    for (int i = threadIdx.x; i < n_ee * n_joints; i += blockDim.x)
        s_ancestor_masks[i] = ancestor_masks[i];

    // Each thread handles one particle
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    // Load initial particle positions into shared memory (before first sync)
    if (t < n_particles) {
        for (int a = 0; a < n_act; a++)
            s_particles[t * n_act + a] = seeds[(p * n_particles + t) * n_act + a];
    }

    __syncthreads();

    if (t >= n_particles) return;

    // Per-thread best-solution tracking
    float best_cfg[MAX_ACT];
    float best_err = 1e30f;
    for (int a = 0; a < n_act; a++)
        best_cfg[a] = s_particles[t * n_act + a];

    // Scratch buffers (thread-private to avoid shared-memory pressure)
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    // ---- Main SVGD loop ----
    for (int iter = 0; iter < n_iters; iter++) {

        // ------------------------------------------------------------------
        // Step 1: compute grad_logp for particle t and store in shared memory.
        //
        // Also compute the per-particle median pairwise distance² for the
        // adaptive bandwidth estimate.  Each thread sorts its own row of
        // distances (N ≤ 32 elements) with a simple insertion sort and writes
        // the median to s_row_med[t].  Thread 0 then finds the median of
        // those N values and updates s_bandwidth.
        // ------------------------------------------------------------------
        float* cfg = s_particles + t * n_act;

        // --- Pairwise distances for bandwidth (insertion-sort row t) ---
        float local_dists[MAX_PARTICLES];
        for (int j = 0; j < n_particles; j++) {
            float* p_j = s_particles + j * n_act;
            float dsq = 0.0f;
            for (int a = 0; a < n_act; a++) {
                float d = cfg[a] - p_j[a];
                dsq += d * d;
            }
            local_dists[j] = dsq;
        }
        // Insertion sort — O(N²) but N ≤ 32 so very cheap
        for (int i = 1; i < n_particles; i++) {
            float key = local_dists[i];
            int k = i - 1;
            while (k >= 0 && local_dists[k] > key) {
                local_dists[k + 1] = local_dists[k];
                --k;
            }
            local_dists[k + 1] = key;
        }
        // After sorting, local_dists[0] == 0 (self-distance).
        // Use index n_particles/2 so the self-distance doesn't dominate.
        s_row_med[t] = local_dists[n_particles / 2];

        // --- Jacobian / residual for grad_logp ---
        compute_multi_ee_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_target_jnts, s_ancestor_masks, s_target_Ts,
            n_joints, n_act, n_ee, r, J);

        // grad_logp = d/dq [-||r||^2] = -2 * J^T * r
        for (int a = 0; a < n_act; a++) {
            float g = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++)
                g += J[k * n_act + a] * r[k];
            s_grad_logp[t * n_act + a] = -2.0f * g;
        }

        // Update best solution
        float curr_err = 0.0f;
        for (int k = 0; k < 6 * n_ee; k++) curr_err += r[k] * r[k];
        if (curr_err < best_err) {
            best_err = curr_err;
            for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];
        }

        __syncthreads();

        // Thread 0: compute adaptive bandwidth from median of row medians.
        // All other threads are idle here, so the in-place sort of s_row_med
        // is safe — no other thread reads it after the sync above.
        if (t == 0) {
            // Insertion sort of n_particles row-medians
            for (int i = 1; i < n_particles; i++) {
                float key = s_row_med[i];
                int k = i - 1;
                while (k >= 0 && s_row_med[k] > key) {
                    s_row_med[k + 1] = s_row_med[k];
                    --k;
                }
                s_row_med[k + 1] = key;
            }
            float med = s_row_med[n_particles / 2];
            float log_n = logf((float)n_particles);
            // bandwidth = sqrt(median_dist² / log N).
            // Fall back to caller-supplied value when particles have collapsed.
            float h = med / (log_n + 1e-8f);
            s_bandwidth = (h > 1e-8f) ? sqrtf(h) : bandwidth;
        }

        __syncthreads();   // all threads wait for s_bandwidth before Step 2

        // ------------------------------------------------------------------
        // Step 2: compute SVGD phi for particle t using all particles
        //
        //   phi(x_i) = (1/N) sum_j [ k(x_j, x_i) * grad_logp(x_j)
        //                           + grad_{x_j} k(x_j, x_i)        ]
        //
        // where:
        //   k(x_j, x_i)              = exp(-||x_j - x_i||^2 / (2h^2))
        //   grad_{x_j} k(x_j, x_i)  = k * (x_i - x_j) / h^2
        // ------------------------------------------------------------------
        const float bw      = s_bandwidth;
        const float bw_sq   = bw * bw + 1e-8f;

        float phi[MAX_ACT];
        for (int a = 0; a < n_act; a++) phi[a] = 0.0f;

        for (int j = 0; j < n_particles; j++) {
            float* p_j     = s_particles + j * n_act;
            float* glogp_j = s_grad_logp + j * n_act;

            // RBF kernel value k(x_j, x_i)
            float dist_sq = 0.0f;
            for (int a = 0; a < n_act; a++) {
                float d = p_j[a] - cfg[a];
                dist_sq += d * d;
            }
            float k_val = expf(-dist_sq / (2.0f * bw_sq));

            for (int a = 0; a < n_act; a++) {
                // k(x_j, x_i) * grad_logp(x_j)
                phi[a] += k_val * glogp_j[a];
                // grad_{x_j} k(x_j, x_i) = k * (x_i - x_j) / h^2
                phi[a] += k_val * (cfg[a] - p_j[a]) / bw_sq;
            }
        }

        // Normalize by N
        float n_inv = 1.0f / (float)n_particles;
        for (int a = 0; a < n_act; a++) phi[a] *= n_inv;

        __syncthreads();

        // ------------------------------------------------------------------
        // Step 3: update shared particle state
        // ------------------------------------------------------------------
        for (int a = 0; a < n_act; a++) {
            if (!s_fixed_mask[a]) {
                s_particles[t * n_act + a] = clampf(
                    s_particles[t * n_act + a] + step_size * phi[a],
                    s_lower[a], s_upper[a]);
            }
        }

        __syncthreads();
    }

    // ---- Write outputs ----
    const int gs = p * n_particles + t;
    for (int a = 0; a < n_act; a++) out[gs * n_act + a] = best_cfg[a];
    out_err[gs] = best_err;

    // FK on best_cfg to get final EE position
    compute_multi_ee_residual_only(
        best_cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r);
    int tgt0 = s_target_jnts[0];
    out_ee[gs * 3 + 0] = T_world[tgt0 * 7 + 4];
    out_ee[gs * 3 + 1] = T_world[tgt0 * 7 + 5];
    out_ee[gs * 3 + 2] = T_world[tgt0 * 7 + 6];
    out_target[gs * 3 + 0] = s_target_Ts[4];
    out_target[gs * 3 + 1] = s_target_Ts[5];
    out_target[gs * 3 + 2] = s_target_Ts[6];
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error SvgdRegionIkCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::S32> target_jnts,
    ffi::Buffer<ffi::DataType::S32> ancestor_masks,
    ffi::Buffer<ffi::DataType::F32> target_Ts,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t n_iters,
    float bandwidth,
    float step_size,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_ee,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_target)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_particles = static_cast<int>(seeds.dimensions()[1]);
    const int n_act = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints = static_cast<int>(twists.dimensions()[0]);
    const int n_ee = static_cast<int>(target_jnts.dimensions()[0]);

    constexpr int THREADS_MAX = 32;
    const int threads = n_particles < THREADS_MAX ? n_particles : THREADS_MAX;
    const int blocks_x = (n_particles + threads - 1) / threads;

    svgd_region_ik_kernel<<<dim3(blocks_x, n_problems), threads, 0, stream>>>(
        seeds.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        target_jnts.typed_data(),
        ancestor_masks.typed_data(),
        target_Ts.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        n_iters,
        bandwidth,
        step_size,
        out->typed_data(),
        out_err->typed_data(),
        out_ee->typed_data(),
        out_target->typed_data(),
        n_problems, n_particles, n_joints, n_act, n_ee);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Handler registration
// ---------------------------------------------------------------------------

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SvgdRegionIkCudaFfi, SvgdRegionIkCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // target_jnts
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // ancestor_masks
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fixed_mask
        .Attr<int64_t>("n_iters")
        .Attr<float>("bandwidth")
        .Attr<float>("step_size")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out cfgs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out errors
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out ee_points
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out target_points
);
