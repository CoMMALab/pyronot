/**
 * STOMP/MPPI TrajOpt CUDA kernel — Multi-kernel architecture.
 *
 * Architecture (per STOMP iteration):
 *   Kernel 1: stomp_eval_cost_kernel  — FK + collision + smoothness + limits per timestep,
 *                                       noise generated inline via FIR RNG replay,
 *                                       parallel reduce over T → per-sample cost
 *   Kernel 2: stomp_softmax_kernel    — parallel softmax → importance weights
 *   Kernel 3: stomp_update_kernel     — parallel weighted noise accumulation + trajectory update,
 *                                       noise regenerated inline (no global noise buffer)
 *
 * Key design decisions:
 *   - No global noise buffer: perturbations are regenerated on-the-fly in both eval and
 *     update kernels from (base_seed, iter, k, t, d).  This eliminates B*T*DOF*K*4 bytes
 *     of global memory traffic per iteration — the dominant bandwidth bottleneck.
 *   - FIR smooth noise: a short (7-tap, σ=2) Gaussian filter applied to independent
 *     normal samples gives O(T) smooth noise instead of the O(T²) Cholesky triangular
 *     solve.  L_inv_T is no longer needed.
 *   - Parallelises over timesteps T (not just samples K)
 *   - Replaces atomicAdd with parallel reductions
 *   - Replaces serial thread-0 softmax with parallel reduction
 *
 * Build with:
 *   bash src/pyroffi/cuda_kernels/build_stomp_trajopt_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "_collision_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits
// ---------------------------------------------------------------------------

#ifndef STOMP_MAX_T
#define STOMP_MAX_T    64
#endif

#ifndef STOMP_MAX_DOF
#define STOMP_MAX_DOF  8
#endif

#ifndef STOMP_MAX_N
#define STOMP_MAX_N    MAX_JOINTS   // 64
#endif

#ifndef STOMP_MAX_S
#define STOMP_MAX_S    8
#endif

#ifndef STOMP_MAX_PAIRS
#define STOMP_MAX_PAIRS 256
#endif

#ifndef STOMP_MAX_K
#define STOMP_MAX_K    512
#endif

// ---------------------------------------------------------------------------
// Parallel reduction helpers (block size must be power of 2)
// ---------------------------------------------------------------------------

static __device__ __forceinline__
void block_reduce_sum(float* smem, int tid)
{
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
}

// Warp-/block-level scalar reductions used by softmax/update kernels.
// These avoid large shared-memory scratch buffers and reduce sync overhead.
static __device__ __forceinline__ float warp_reduce_sum_scalar(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static __device__ __forceinline__ float warp_reduce_min_scalar(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fminf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

static __device__ __forceinline__ float block_reduce_sum_scalar(float v)
{
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;

    v = warp_reduce_sum_scalar(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    float out = (threadIdx.x < nwarp) ? warp_sums[lane] : 0.0f;
    if (wid == 0) out = warp_reduce_sum_scalar(out);
    __syncthreads();
    return out;
}

static __device__ __forceinline__ float block_reduce_min_scalar(float v)
{
    __shared__ float warp_mins[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;

    v = warp_reduce_min_scalar(v);
    if (lane == 0) warp_mins[wid] = v;
    __syncthreads();

    float out = (threadIdx.x < nwarp) ? warp_mins[lane] : 1e30f;
    if (wid == 0) out = warp_reduce_min_scalar(out);
    __syncthreads();
    return out;
}

// ---------------------------------------------------------------------------
// Geometry distance primitives (shared)
// ---------------------------------------------------------------------------

#define stomp_sphere_sphere_dist sphere_sphere_dist
#define stomp_sphere_capsule_dist sphere_capsule_dist
#define stomp_sphere_box_dist sphere_box_dist
#define stomp_sphere_halfspace_dist sphere_halfspace_dist
#define stomp_apply_se3 apply_se3_point

// ---------------------------------------------------------------------------
// Per-configuration collision hinge cost  (full nonlinear, no Jacobian)
// ---------------------------------------------------------------------------

static __device__ float stomp_collision_cost_cfg(
    const float* __restrict__ cfg,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const float* __restrict__ sphere_off,
    const float* __restrict__ sphere_rad,
    const int*   __restrict__ pair_i,
    const int*   __restrict__ pair_j,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    int n_joints, int n_act, int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    float collision_margin,
    float* __restrict__ T_world)   // thread-private FK workspace [n_joints*7]
{
    fk_single(cfg, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv,
              T_world, n_joints, n_act);

    float cost = 0.0f;

    // Self-collision
    for (int p = 0; p < P; p++) {
        int li = pair_i[p], lj = pair_j[p];
        float min_d = 1e10f;
        for (int si = 0; si < S; si++) {
            float ri = sphere_rad[li*S+si];
            if (ri < 0.0f) continue;
            float lci[3] = {sphere_off[(li*S+si)*3],
                            sphere_off[(li*S+si)*3+1],
                            sphere_off[(li*S+si)*3+2]};
            float ci[3]; stomp_apply_se3(T_world + li*7, lci, ci);
            for (int sj = 0; sj < S; sj++) {
                float rj = sphere_rad[lj*S+sj];
                if (rj < 0.0f) continue;
                float lcj[3] = {sphere_off[(lj*S+sj)*3],
                                sphere_off[(lj*S+sj)*3+1],
                                sphere_off[(lj*S+sj)*3+2]};
                float cj[3]; stomp_apply_se3(T_world + lj*7, lcj, cj);
                float d = stomp_sphere_sphere_dist(ci[0],ci[1],ci[2],ri,
                                                   cj[0],cj[1],cj[2],rj);
                if (d < min_d) min_d = d;
            }
        }
        if (min_d < 1e9f) {
            float v = fmaxf(0.0f, collision_margin - min_d);
            cost += v * v;
        }
    }

    // World obstacles
    for (int j = 0; j < N; j++) {
        for (int m = 0; m < Ms; m++) {
            float min_d = 1e10f;
            for (int s = 0; s < S; s++) {
                float r = sphere_rad[j*S+s];
                if (r < 0.0f) continue;
                float lc[3] = {sphere_off[(j*S+s)*3], sphere_off[(j*S+s)*3+1], sphere_off[(j*S+s)*3+2]};
                float wc[3]; stomp_apply_se3(T_world+j*7, lc, wc);
                float d = stomp_sphere_sphere_dist(wc[0],wc[1],wc[2],r,
                    world_spheres[m*4],world_spheres[m*4+1],
                    world_spheres[m*4+2],world_spheres[m*4+3]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) { float v=fmaxf(0.f,collision_margin-min_d); cost+=v*v; }
        }
        for (int m = 0; m < Mc; m++) {
            float min_d = 1e10f;
            const float* cap = world_capsules + m*7;
            for (int s = 0; s < S; s++) {
                float r = sphere_rad[j*S+s];
                if (r < 0.0f) continue;
                float lc[3] = {sphere_off[(j*S+s)*3], sphere_off[(j*S+s)*3+1], sphere_off[(j*S+s)*3+2]};
                float wc[3]; stomp_apply_se3(T_world+j*7, lc, wc);
                float d = stomp_sphere_capsule_dist(wc[0],wc[1],wc[2],r,
                    cap[0],cap[1],cap[2],cap[3],cap[4],cap[5],cap[6]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) { float v=fmaxf(0.f,collision_margin-min_d); cost+=v*v; }
        }
        for (int m = 0; m < Mb; m++) {
            float min_d = 1e10f;
            const float* bx = world_boxes + m*15;
            for (int s = 0; s < S; s++) {
                float r = sphere_rad[j*S+s];
                if (r < 0.0f) continue;
                float lc[3] = {sphere_off[(j*S+s)*3], sphere_off[(j*S+s)*3+1], sphere_off[(j*S+s)*3+2]};
                float wc[3]; stomp_apply_se3(T_world+j*7, lc, wc);
                float d = stomp_sphere_box_dist(wc[0],wc[1],wc[2],r,
                    bx[0],bx[1],bx[2], bx[3],bx[4],bx[5],
                    bx[6],bx[7],bx[8], bx[9],bx[10],bx[11],
                    bx[12],bx[13],bx[14]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) { float v=fmaxf(0.f,collision_margin-min_d); cost+=v*v; }
        }
        for (int m = 0; m < Mh; m++) {
            float min_d = 1e10f;
            const float* hs = world_halfspaces + m*6;
            for (int s = 0; s < S; s++) {
                float r = sphere_rad[j*S+s];
                if (r < 0.0f) continue;
                float lc[3] = {sphere_off[(j*S+s)*3], sphere_off[(j*S+s)*3+1], sphere_off[(j*S+s)*3+2]};
                float wc[3]; stomp_apply_se3(T_world+j*7, lc, wc);
                float d = stomp_sphere_halfspace_dist(wc[0],wc[1],wc[2],r,
                    hs[0],hs[1],hs[2],hs[3],hs[4],hs[5]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) { float v=fmaxf(0.f,collision_margin-min_d); cost+=v*v; }
        }
    }

    return cost;
}

// ---------------------------------------------------------------------------
// Deterministic RNG: per-sample noise from (block_seed, iter, k, t, d)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float stomp_noise(
    uint64_t base_seed,
    int iter, int k, int t, int d,
    float noise_scale)
{
    uint32_t state = (uint32_t)(base_seed
        ^ ((uint32_t)iter  * 2654435761u)
        ^ ((uint32_t)k     * 2246822519u)
        ^ ((uint32_t)t     * 3266489917u)
        ^ ((uint32_t)d     *    1664525u));
    if (state == 0u) state = 1u;
    return rng_normal(state) * noise_scale;
}

// ---------------------------------------------------------------------------
// Inline FIR smooth noise — O(T) alternative to O(T²) Cholesky triangular solve.
//
// A 7-tap Gaussian filter (half_width=3, σ=2.0) applied to independent N(0,1)
// draws gives smooth correlated samples without any global noise buffer.
// The raw samples are scaled by INV_GAIN so that the filtered output has the
// desired noise_scale std (gain = sqrt(sum(kernel²))).
//
// Boundary: t=0 and t=T-1 are pinned endpoints — always return 0.
// Neighbouring draws that fall outside [1, T-2] are treated as 0 (zero padding).
// ---------------------------------------------------------------------------

#define FIR_HALF_WIDTH 3

// Normalised Gaussian kernel: exp(-0.5*(x/2)²) / sum, x in [-3..3]
static __device__ __constant__ float FIR_KERNEL[7] = {
    0.07015f, 0.13107f, 0.19071f, 0.21608f, 0.19071f, 0.13107f, 0.07015f
};
// 1 / sqrt(sum(FIR_KERNEL²))  so that filtered output std == raw input std
#define FIR_INV_GAIN 2.4721f

static __device__ __forceinline__ float stomp_smooth_noise_fir(
    uint64_t block_seed, int iter, int k, int t, int d, int T,
    float noise_scale)
{
    if (k == 0 || t == 0 || t >= T - 1) return 0.0f;

    const float scaled = noise_scale * FIR_INV_GAIN;
    float eps = 0.0f;
    for (int j = -FIR_HALF_WIDTH; j <= FIR_HALF_WIDTH; j++) {
        int tj = t + j;
        float z_j = (tj > 0 && tj < T - 1)
                    ? stomp_noise(block_seed, iter, k, tj, d, scaled)
                    : 0.0f;
        eps += FIR_KERNEL[j + FIR_HALF_WIDTH] * z_j;
    }
    return eps;
}

// =========================================================================
// KERNEL 1: Cost evaluation — FK + collision + smoothness + limits
// =========================================================================
//
// Grid:  (B, K)            — one block per (batch, sample) pair
// Block: next_pow2(T)      — one thread per timestep (excess threads idle)
//
// Shared memory: s_q[T * n_act] + s_reduce[blockDim.x]
//
// Each thread computes FK + collision + limits cost for its timestep,
// plus smoothness stencil contributions.  A parallel reduction sums
// per-timestep costs → per-sample cost in d_costs[B, K].
// Perturbations are generated inline via stomp_smooth_noise_fir — no global
// noise buffer needed.

__global__ void stomp_eval_cost_kernel(
    const float* __restrict__ traj,     // [B, T, n_act]
    float*       __restrict__ costs,    // [B, K]
    uint64_t base_seed, int iter, float noise_scale,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const float* __restrict__ sphere_off,
    const float* __restrict__ sphere_rad,
    const int*   __restrict__ pair_i_buf,
    const int*   __restrict__ pair_j_buf,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    int T, int n_joints, int n_act, int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh, int K,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_coll, float collision_margin)
{
    const int b = blockIdx.x;
    const int k = blockIdx.y;
    const int t = threadIdx.x;

    extern __shared__ float smem[];
    float* s_q      = smem;                   // [T * n_act]
    float* s_reduce = smem + T * n_act;       // [blockDim.x]

    float my_cost = 0.0f;

    const uint64_t block_seed = base_seed ^ ((uint64_t)b * 6364136223846793005ULL);

    if (t < T) {
        // Load perturbed config into shared memory — noise generated inline (no buffer)
        for (int d = 0; d < n_act; d++) {
            float base_val = traj[b * T * n_act + t * n_act + d];
            float n_val = stomp_smooth_noise_fir(block_seed, iter, k, t, d, T, noise_scale);
            s_q[t * n_act + d] = base_val + n_val;
        }
    }
    __syncthreads();

    if (t < T) {
        // ── Smoothness (4th-order stencil: acceleration + jerk) ──
        const float ST[5] = {-1.f/12.f, 16.f/12.f, -30.f/12.f, 16.f/12.f, -1.f/12.f};

        // Acceleration cost: stencil at [t .. t+4], valid for t < T-4
        if (t < T - 4) {
            for (int d = 0; d < n_act; d++) {
                float a = 0.0f;
                for (int i = 0; i < 5; i++)
                    a += ST[i] * s_q[(t + i) * n_act + d];
                my_cost += w_smooth * w_acc * a * a;
            }
        }

        // Jerk cost: difference of consecutive acceleration stencils
        if (t < T - 5) {
            for (int d = 0; d < n_act; d++) {
                float a0 = 0.0f, a1 = 0.0f;
                for (int i = 0; i < 5; i++) {
                    a0 += ST[i] * s_q[(t + i) * n_act + d];
                    a1 += ST[i] * s_q[(t + 1 + i) * n_act + d];
                }
                float j = a1 - a0;
                my_cost += w_smooth * w_jerk * j * j;
            }
        }

        // ── Limits cost ──
        float q_t[STOMP_MAX_DOF];
        for (int d = 0; d < n_act; d++)
            q_t[d] = s_q[t * n_act + d];

        for (int d = 0; d < n_act; d++) {
            float viol = fmaxf(0.f, q_t[d] - upper[d])
                       + fmaxf(0.f, lower[d] - q_t[d]);
            my_cost += w_limits * viol * viol;
        }

        // ── FK + Collision cost ──
        float T_world[STOMP_MAX_N * 7];
        my_cost += w_coll * stomp_collision_cost_cfg(
            q_t, twists, parent_tf, parent_idx, act_idx,
            mimic_mul, mimic_off, mimic_act_idx, topo_inv,
            sphere_off, sphere_rad, pair_i_buf, pair_j_buf,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
            n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
            collision_margin, T_world);
    }

    // ── Parallel reduction over timesteps → per-sample cost ──
    s_reduce[threadIdx.x] = my_cost;
    __syncthreads();
    block_reduce_sum(s_reduce, threadIdx.x);

    if (threadIdx.x == 0)
        costs[b * K + k] = s_reduce[0];
}

// =========================================================================
// KERNEL 2: Softmax importance weights (parallel)
// =========================================================================
//
// Grid:  (B)              — one block per batch trajectory
// Block: next_pow2(K)     — one thread per sample

__global__ void stomp_softmax_kernel(
    const float* __restrict__ costs,    // [B, K]
    float*       __restrict__ weights,  // [B, K]
    int K, float temperature)
{
    const int b = blockIdx.x;
    const int k = threadIdx.x;

    // Per-trajectory softmax: one block owns one trajectory b.
    const float c = (k < K) ? costs[b * K + k] : 1e30f;
    const float min_c = block_reduce_min_scalar(c);
    const float shifted = (k < K) ? (c - min_c) : 0.0f;

    const float sum_shift = block_reduce_sum_scalar((k < K) ? shifted : 0.0f);
    const float mean_shift = sum_shift / (float)K;

    const float diff = shifted - mean_shift;
    const float sum_sq = block_reduce_sum_scalar((k < K) ? (diff * diff) : 0.0f);
    const float std_shift = sqrtf(sum_sq / (float)K);
    const float beta = fmaxf(std_shift, 1e-6f) * temperature;

    const float w = (k < K) ? expf(-shifted / (beta + 1e-18f)) : 0.0f;
    const float sum_w = block_reduce_sum_scalar((k < K) ? w : 0.0f);

    if (k < K) {
        weights[b * K + k] = w / (sum_w + 1e-30f);
    }
}

// =========================================================================
// KERNEL 3: Weighted trajectory update (no atomics)
// =========================================================================
//
// Grid:  (B, T-2)         — one block per (batch, interior timestep)
// Block: next_pow2(K)     — one thread per sample
//
// For each DOF, each thread k regenerates weighted noise inline and a block-level
// warp reduction sums across K → delta[d]. Thread 0 applies the trajectory update.

__global__ void stomp_update_kernel(
    float*       __restrict__ traj,     // [B, T, n_act] — updated in-place
    const float* __restrict__ weights,  // [B, K]
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    uint64_t base_seed, int iter, float noise_scale,
    int K, int T, int n_act, float step_size)
{
    const int b  = blockIdx.x;
    const int t  = blockIdx.y + 1;   // interior timestep: 1 .. T-2
    const int k  = threadIdx.x;

    const uint64_t block_seed = base_seed ^ ((uint64_t)b * 6364136223846793005ULL);

    const float w = (k < K) ? weights[b * K + k] : 0.0f;

    // Regenerate weighted noise inline and reduce over K independently per DOF.
    for (int d = 0; d < n_act; d++) {
        float n_val = (k < K)
            ? stomp_smooth_noise_fir(block_seed, iter, k, t, d, T, noise_scale)
            : 0.0f;
        float delta_d = block_reduce_sum_scalar((k < K) ? (w * n_val) : 0.0f);

        if (k == 0) {
            const int idx = b * T * n_act + t * n_act + d;
            const float val = traj[idx] + step_size * delta_d;
            traj[idx] = fmaxf(lower[d], fminf(upper[d], val));
        }
    }
}

// =========================================================================
// KERNEL 4: Track best sampled trajectory (one block per trajectory)
// =========================================================================
//
// Grid:  (B)              — one block per trajectory
// Block: next_pow2(K)     — reduce over K sampled costs
//
// Finds argmin_k cost[b,k], and if improved over best_cost[b], writes the
// corresponding sampled trajectory (traj + regenerated smooth noise for k*)
// to best_traj[b].

__global__ void stomp_track_best_kernel(
    const float* __restrict__ traj,      // [B, T, n_act]
    const float* __restrict__ costs,     // [B, K]
    float*       __restrict__ best_traj, // [B, T, n_act]
    float*       __restrict__ best_cost, // [B]
    uint64_t base_seed, int iter, float noise_scale,
    int B, int K, int T, int n_act)
{
    const int b = blockIdx.x;
    const int k = threadIdx.x;

    extern __shared__ float smem[];
    float* s_cost = smem;
    float* s_idxf = smem + blockDim.x;

    float c = (k < K) ? costs[b * K + k] : 1e30f;
    float idxf = (float)k;
    s_cost[k] = c;
    s_idxf[k] = idxf;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (k < s) {
            float c2 = s_cost[k + s];
            if (c2 < s_cost[k]) {
                s_cost[k] = c2;
                s_idxf[k] = s_idxf[k + s];
            }
        }
        __syncthreads();
    }

    if (k == 0) {
        const float iter_best_cost = s_cost[0];
        if (iter_best_cost < best_cost[b]) {
            const int k_best = (int)s_idxf[0];
            const uint64_t block_seed = base_seed ^ ((uint64_t)b * 6364136223846793005ULL);

            for (int t = 0; t < T; t++) {
                for (int d = 0; d < n_act; d++) {
                    const float base_val = traj[b * T * n_act + t * n_act + d];
                    const float n_val = stomp_smooth_noise_fir(
                        block_seed, iter, k_best, t, d, T, noise_scale);
                    best_traj[b * T * n_act + t * n_act + d] = base_val + n_val;
                }
            }
            best_cost[b] = iter_best_cost;
        }
    }
}

// =========================================================================
// KERNEL 5: Final cost evaluation (no noise)
// =========================================================================
//
// Grid:  (B)              — one block per batch trajectory
// Block: next_pow2(T)     — one thread per timestep
//
// Identical to eval_cost_kernel but reads trajectory directly (no noise).

__global__ void stomp_final_cost_kernel(
    const float* __restrict__ traj,
    float*       __restrict__ costs,    // [B]
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const float* __restrict__ sphere_off,
    const float* __restrict__ sphere_rad,
    const int*   __restrict__ pair_i_buf,
    const int*   __restrict__ pair_j_buf,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    int T, int n_joints, int n_act, int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_coll, float collision_margin)
{
    const int b = blockIdx.x;
    const int t = threadIdx.x;

    extern __shared__ float smem[];
    float* s_q      = smem;                   // [T * n_act]
    float* s_reduce = smem + T * n_act;       // [blockDim.x]

    float my_cost = 0.0f;

    if (t < T) {
        for (int d = 0; d < n_act; d++)
            s_q[t * n_act + d] = traj[b * T * n_act + t * n_act + d];
    }
    __syncthreads();

    if (t < T) {
        const float ST[5] = {-1.f/12.f, 16.f/12.f, -30.f/12.f, 16.f/12.f, -1.f/12.f};

        if (t < T - 4) {
            for (int d = 0; d < n_act; d++) {
                float a = 0.0f;
                for (int i = 0; i < 5; i++)
                    a += ST[i] * s_q[(t + i) * n_act + d];
                my_cost += w_smooth * w_acc * a * a;
            }
        }

        if (t < T - 5) {
            for (int d = 0; d < n_act; d++) {
                float a0 = 0.0f, a1 = 0.0f;
                for (int i = 0; i < 5; i++) {
                    a0 += ST[i] * s_q[(t + i) * n_act + d];
                    a1 += ST[i] * s_q[(t + 1 + i) * n_act + d];
                }
                float j = a1 - a0;
                my_cost += w_smooth * w_jerk * j * j;
            }
        }

        float q_t[STOMP_MAX_DOF];
        for (int d = 0; d < n_act; d++)
            q_t[d] = s_q[t * n_act + d];

        for (int d = 0; d < n_act; d++) {
            float viol = fmaxf(0.f, q_t[d] - upper[d])
                       + fmaxf(0.f, lower[d] - q_t[d]);
            my_cost += w_limits * viol * viol;
        }

        float T_world[STOMP_MAX_N * 7];
        my_cost += w_coll * stomp_collision_cost_cfg(
            q_t, twists, parent_tf, parent_idx, act_idx,
            mimic_mul, mimic_off, mimic_act_idx, topo_inv,
            sphere_off, sphere_rad, pair_i_buf, pair_j_buf,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
            n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
            collision_margin, T_world);
    }

    s_reduce[threadIdx.x] = my_cost;
    __syncthreads();
    block_reduce_sum(s_reduce, threadIdx.x);

    if (threadIdx.x == 0)
        costs[b] = s_reduce[0];
}

__global__ void stomp_init_best_cost_kernel(
    float* __restrict__ best_cost,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) best_cost[i] = 1e30f;
}

// ---------------------------------------------------------------------------
// Host helper: next power of 2
// ---------------------------------------------------------------------------

static inline int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ---------------------------------------------------------------------------
// XLA FFI handler — orchestrates multi-kernel STOMP loop
// ---------------------------------------------------------------------------

static ffi::Error StompTrajoptImpl(
    cudaStream_t                      stream,
    ffi::Buffer<ffi::DataType::F32>   init_trajs,
    ffi::Buffer<ffi::DataType::F32>   twists,
    ffi::Buffer<ffi::DataType::F32>   parent_tf,
    ffi::Buffer<ffi::DataType::S32>   parent_idx,
    ffi::Buffer<ffi::DataType::S32>   act_idx,
    ffi::Buffer<ffi::DataType::F32>   mimic_mul,
    ffi::Buffer<ffi::DataType::F32>   mimic_off,
    ffi::Buffer<ffi::DataType::S32>   mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32>   topo_inv,
    ffi::Buffer<ffi::DataType::F32>   sphere_off,
    ffi::Buffer<ffi::DataType::F32>   sphere_rad,
    ffi::Buffer<ffi::DataType::S32>   pair_i,
    ffi::Buffer<ffi::DataType::S32>   pair_j,
    ffi::Buffer<ffi::DataType::F32>   world_spheres,
    ffi::Buffer<ffi::DataType::F32>   world_capsules,
    ffi::Buffer<ffi::DataType::F32>   world_boxes,
    ffi::Buffer<ffi::DataType::F32>   world_halfspaces,
    ffi::Buffer<ffi::DataType::F32>   lower,
    ffi::Buffer<ffi::DataType::F32>   upper,
    ffi::Buffer<ffi::DataType::F32>   start,
    ffi::Buffer<ffi::DataType::F32>   goal,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_trajs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_costs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> _workspace,
    // Attributes
    int64_t n_iters,
    int64_t n_samples,
    int64_t S,
    float   noise_scale,
    float   temperature,
    float   step_size,
    float   w_smooth,
    float   w_acc,
    float   w_jerk,
    float   w_limits,
    float   w_collision,
    float   w_collision_max,
    float   collision_penalty_scale,
    float   collision_margin,
    int64_t rng_seed)
{
    // ── Extract dimensions ──
    auto shape = init_trajs.dimensions();
    int B      = (int)shape[0];
    int T      = (int)shape[1];
    int n_act  = (int)shape[2];

    int n_joints = (int)twists.dimensions()[0];
    int N        = (int)sphere_rad.dimensions()[0] / (int)S;
    int P        = (int)pair_i.dimensions()[0];
    int Ms       = (int)world_spheres.dimensions()[0];
    int Mc       = (int)world_capsules.dimensions()[0];
    int Mb       = (int)world_boxes.dimensions()[0];
    int Mh       = (int)world_halfspaces.dimensions()[0];

    if (B == 0 || T == 0 || n_act == 0)
        return ffi::Error::Success();

    int K = (int)n_samples;
    if (K > STOMP_MAX_K)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "n_samples exceeds STOMP_MAX_K");
    if (T > STOMP_MAX_T)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "T exceeds STOMP_MAX_T");
    if (n_act > STOMP_MAX_DOF)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "n_act exceeds STOMP_MAX_DOF");

    // ── Partition workspace buffer ──
    // Layout: best_trajs[B*T*n_act] | best_costs[B] | costs[B*K] | weights[B*K]
    float* d_workspace = _workspace->typed_data();
    float* d_best_traj = d_workspace;
    float* d_best_cost = d_best_traj + B * T * n_act;
    float* d_costs     = d_best_cost + B;
    float* d_weights   = d_costs + B * K;

    // ── Copy initial trajectories to output (updated in-place) ──
    float* d_traj = out_trajs->typed_data();
    cudaMemcpyAsync(d_traj, init_trajs.typed_data(),
                    B * T * n_act * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // Initialize best trajectory/cost buffers.
    cudaMemcpyAsync(d_best_traj, d_traj,
                    B * T * n_act * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    const int init_blocks = (B + 255) / 256;
    stomp_init_best_cost_kernel<<<init_blocks, 256, 0, stream>>>(d_best_cost, B);

    // ── Precompute block sizes (all power of 2) ──
    int T_block = next_pow2(T);    // for eval/final cost kernels
    int K_block = next_pow2(K);    // for softmax/update kernels

    // ── Eval cost kernel config ──
    int eval_smem = (T * n_act + T_block) * (int)sizeof(float);

    // ── Per-iteration constant pointers ──
    const float* d_twists     = twists.typed_data();
    const float* d_parent_tf  = parent_tf.typed_data();
    const int*   d_parent_idx = parent_idx.typed_data();
    const int*   d_act_idx    = act_idx.typed_data();
    const float* d_mimic_mul  = mimic_mul.typed_data();
    const float* d_mimic_off  = mimic_off.typed_data();
    const int*   d_mimic_act  = mimic_act_idx.typed_data();
    const int*   d_topo_inv   = topo_inv.typed_data();
    const float* d_sphere_off = sphere_off.typed_data();
    const float* d_sphere_rad = sphere_rad.typed_data();
    const int*   d_pair_i     = pair_i.typed_data();
    const int*   d_pair_j     = pair_j.typed_data();
    const float* d_world_s    = world_spheres.typed_data();
    const float* d_world_c    = world_capsules.typed_data();
    const float* d_world_b    = world_boxes.typed_data();
    const float* d_world_h    = world_halfspaces.typed_data();
    const float* d_lower      = lower.typed_data();
    const float* d_upper      = upper.typed_data();

    uint64_t base_seed = (uint64_t)rng_seed;
    float w_coll = w_collision;

    // ═══════════════════════════════════════════════════════════════════════
    // Main STOMP loop — 3 kernels per iteration on the same stream
    // ═══════════════════════════════════════════════════════════════════════

    for (int iter = 0; iter < (int)n_iters; iter++) {

        // ── Kernel 1: Evaluate per-sample cost (noise generated inline) ──
        stomp_eval_cost_kernel<<<dim3(B, K), T_block, eval_smem, stream>>>(
            d_traj, d_costs,
            base_seed, iter, noise_scale,
            d_twists, d_parent_tf, d_parent_idx, d_act_idx,
            d_mimic_mul, d_mimic_off, d_mimic_act, d_topo_inv,
            d_sphere_off, d_sphere_rad, d_pair_i, d_pair_j,
            d_world_s, d_world_c, d_world_b, d_world_h,
            d_lower, d_upper,
            T, n_joints, n_act, N, (int)S, P,
            Ms, Mc, Mb, Mh, K,
            w_smooth, w_acc, w_jerk, w_limits, w_coll, collision_margin);

        // ── Kernel 2: Softmax importance weights ──
        stomp_softmax_kernel<<<B, K_block, 0, stream>>>(
            d_costs, d_weights, K, temperature);

        // ── Kernel 3: Weighted trajectory update (noise regenerated inline) ──
        stomp_update_kernel<<<dim3(B, T - 2), K_block, 0, stream>>>(
            d_traj, d_weights,
            d_lower, d_upper,
            base_seed, iter, noise_scale,
            K, T, n_act, step_size);

        // ── Kernel 4: Track best sampled candidate for each trajectory ──
        stomp_track_best_kernel<<<B, K_block, 2 * K_block * (int)sizeof(float), stream>>>(
            d_traj, d_costs, d_best_traj, d_best_cost,
            base_seed, iter, noise_scale,
            B, K, T, n_act);

        // Scale collision penalty
        w_coll = fminf(w_coll * collision_penalty_scale, w_collision_max);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Final cost evaluation at w_collision_max
    // ═══════════════════════════════════════════════════════════════════════

    // Return best trajectory found during MPPI iterations.
    cudaMemcpyAsync(d_traj, d_best_traj,
                    B * T * n_act * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    int final_smem = (T * n_act + T_block) * (int)sizeof(float);
    stomp_final_cost_kernel<<<B, T_block, final_smem, stream>>>(
        d_traj, out_costs->typed_data(),
        d_twists, d_parent_tf, d_parent_idx, d_act_idx,
        d_mimic_mul, d_mimic_off, d_mimic_act, d_topo_inv,
        d_sphere_off, d_sphere_rad, d_pair_i, d_pair_j,
        d_world_s, d_world_c, d_world_b, d_world_h,
        d_lower, d_upper,
        T, n_joints, n_act, N, (int)S, P,
        Ms, Mc, Mb, Mh,
        w_smooth, w_acc, w_jerk, w_limits, w_collision_max, collision_margin);

    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// XLA FFI registration
// ---------------------------------------------------------------------------

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StompTrajoptCudaFfi,
    StompTrajoptImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // init_trajs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // sphere_off
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // sphere_rad
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // pair_j
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // start
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // goal
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_trajs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_costs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // workspace
        .Attr<int64_t>("n_iters")
        .Attr<int64_t>("n_samples")
        .Attr<int64_t>("S")
        .Attr<float>("noise_scale")
        .Attr<float>("temperature")
        .Attr<float>("step_size")
        .Attr<float>("w_smooth")
        .Attr<float>("w_acc")
        .Attr<float>("w_jerk")
        .Attr<float>("w_limits")
        .Attr<float>("w_collision")
        .Attr<float>("w_collision_max")
        .Attr<float>("collision_penalty_scale")
        .Attr<float>("collision_margin")
        .Attr<int64_t>("rng_seed")
);
