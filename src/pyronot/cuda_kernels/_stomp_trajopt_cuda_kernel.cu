/**
 * STOMP/MPPI TrajOpt CUDA kernel — Multi-kernel architecture.
 *
 * Architecture (per STOMP iteration):
 *   Kernel 1: stomp_smooth_noise_kernel  — generate smooth noise into global buffer
 *   Kernel 2: stomp_eval_cost_kernel     — FK + collision + smoothness + limits per timestep,
 *                                          parallel reduce over T → per-sample cost
 *   Kernel 3: stomp_softmax_kernel       — parallel softmax → importance weights
 *   Kernel 4: stomp_update_kernel        — parallel weighted noise accumulation + trajectory update
 *
 * Key improvements over the monolithic single-kernel design:
 *   - Parallelises over timesteps T (not just samples K)
 *   - Eliminates per-thread q_k[T][DOF] storage (was the #1 register-spill bottleneck)
 *   - Replaces atomicAdd with parallel reductions
 *   - Replaces serial thread-0 softmax with parallel reduction
 *
 * Noise memory layout: [B, T, n_act, K]
 *   Chosen so the update kernel reads noise for all K samples at a given (t,d)
 *   with stride-1 access → perfect coalescing.
 *
 * Build with:
 *   bash src/pyronot/cuda_kernels/build_stomp_trajopt_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
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
// Noise buffer indexing: layout [B, T, n_act, K]
// ---------------------------------------------------------------------------

#define NOISE_IDX(b, t, d, k, T, D, K) \
    ((b)*(T)*(D)*(K) + (t)*(D)*(K) + (d)*(K) + (k))

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

static __device__ __forceinline__
void block_reduce_min(float* smem, int tid)
{
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fminf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Geometry distance primitives (identical to CHOMP/SCO helpers)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float stomp_sphere_sphere_dist(
    float ax, float ay, float az, float ar,
    float bx, float by, float bz, float br)
{
    float dx = ax - bx, dy = ay - by, dz = az - bz;
    return sqrtf(dx*dx + dy*dy + dz*dz) - (ar + br);
}

__device__ __forceinline__ float stomp_sphere_capsule_dist(
    float sx, float sy, float sz, float sr,
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr)
{
    float vx = x2-x1, vy = y2-y1, vz = z2-z1;
    float len2 = vx*vx + vy*vy + vz*vz;
    float t = 0.0f;
    if (len2 > 1e-12f) {
        t = ((sx-x1)*vx + (sy-y1)*vy + (sz-z1)*vz) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    float cx = x1+t*vx, cy = y1+t*vy, cz = z1+t*vz;
    float dx = sx-cx, dy = sy-cy, dz = sz-cz;
    return sqrtf(dx*dx + dy*dy + dz*dz) - (sr + cr);
}

__device__ __forceinline__ float stomp_box_sdf_local(
    float p1, float p2, float p3,
    float hl1, float hl2, float hl3)
{
    float q1 = fabsf(p1)-hl1, q2 = fabsf(p2)-hl2, q3 = fabsf(p3)-hl3;
    float mq1 = fmaxf(q1,0.f), mq2 = fmaxf(q2,0.f), mq3 = fmaxf(q3,0.f);
    return sqrtf(mq1*mq1+mq2*mq2+mq3*mq3) + fminf(fmaxf(fmaxf(q1,q2),q3),0.f);
}

__device__ __forceinline__ float stomp_sphere_box_dist(
    float sx, float sy, float sz, float sr,
    float bcx, float bcy, float bcz,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float a3x, float a3y, float a3z,
    float hl1, float hl2, float hl3)
{
    float dx = sx-bcx, dy = sy-bcy, dz = sz-bcz;
    float p1 = dx*a1x+dy*a1y+dz*a1z;
    float p2 = dx*a2x+dy*a2y+dz*a2z;
    float p3 = dx*a3x+dy*a3y+dz*a3z;
    return stomp_box_sdf_local(p1, p2, p3, hl1, hl2, hl3) - sr;
}

__device__ __forceinline__ float stomp_sphere_halfspace_dist(
    float sx, float sy, float sz, float sr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    return (sx-px)*nx + (sy-py)*ny + (sz-pz)*nz - sr;
}

__device__ __forceinline__ void stomp_apply_se3(
    const float* __restrict__ T, const float* __restrict__ p,
    float* __restrict__ out)
{
    quat_rotate(T, p, out);
    out[0] += T[4]; out[1] += T[5]; out[2] += T[6];
}

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

// =========================================================================
// KERNEL 1: Smooth noise generation
// =========================================================================
//
// Grid:  (B * K)           — one block per (batch, sample) pair
// Block: (T_in * n_act)    — one thread per (interior-timestep, DOF) pair
//
// Shared memory: s_z[n_act * T_in] + s_L[T_in * T_in]
//
// Generates eps = L_inv_T @ z where z ~ N(0, noise_scale² I)
// and stores in noise[B, T, n_act, K].

__global__ void stomp_smooth_noise_kernel(
    float*       __restrict__ noise,     // [B, T, n_act, K]
    const float* __restrict__ L_inv_T,   // [T_in, T_in] upper triangular
    int B, int K, int T, int n_act, int T_in,
    float noise_scale, uint64_t base_seed, int iter)
{
    const int bk  = blockIdx.x;
    const int b   = bk / K;
    const int k   = bk % K;
    const int tid = threadIdx.x;
    const int d   = tid / T_in;
    const int i   = tid % T_in;

    if (d >= n_act || i >= T_in) return;

    const uint64_t block_seed = base_seed ^ ((uint64_t)b * 6364136223846793005ULL);

    extern __shared__ float smem[];
    float* s_z = smem;                     // [n_act * T_in]
    float* s_L = smem + n_act * T_in;     // [T_in * T_in]

    // Cooperatively load L_inv_T into shared memory
    const int L_size = T_in * T_in;
    for (int idx = tid; idx < L_size; idx += blockDim.x)
        s_L[idx] = L_inv_T[idx];

    // Generate z for this (b, k, d, i)
    float z_val = (k == 0) ? 0.0f
                            : stomp_noise(block_seed, iter, k, i, d, noise_scale);
    s_z[d * T_in + i] = z_val;
    __syncthreads();

    // eps_i = sum_{j>=i} L_inv_T[i, j] * z[d][j]  (upper triangular)
    float eps_i = 0.0f;
    for (int j = i; j < T_in; j++)
        eps_i += s_L[i * T_in + j] * s_z[d * T_in + j];

    // Write to noise[b, i+1, d, k] (i+1 maps interior index to timestep)
    noise[NOISE_IDX(b, i + 1, d, k, T, n_act, K)] = eps_i;

    // Zero endpoints (thread 0 handles once per block)
    if (tid == 0) {
        for (int dd = 0; dd < n_act; dd++) {
            noise[NOISE_IDX(b, 0,   dd, k, T, n_act, K)] = 0.0f;
            noise[NOISE_IDX(b, T-1, dd, k, T, n_act, K)] = 0.0f;
        }
    }
}

// =========================================================================
// KERNEL 2: Cost evaluation — FK + collision + smoothness + limits
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

__global__ void stomp_eval_cost_kernel(
    const float* __restrict__ traj,     // [B, T, n_act]
    const float* __restrict__ noise,    // [B, T, n_act, K]
    float*       __restrict__ costs,    // [B, K]
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

    if (t < T) {
        // Load perturbed config into shared memory
        for (int d = 0; d < n_act; d++) {
            float base_val = traj[b * T * n_act + t * n_act + d];
            float n_val = noise[NOISE_IDX(b, t, d, k, T, n_act, K)];
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
// KERNEL 3: Softmax importance weights (parallel)
// =========================================================================
//
// Grid:  (B)              — one block per batch trajectory
// Block: next_pow2(K)     — one thread per sample
//
// Shared memory: [blockDim.x] floats (reused across reduction phases)

__global__ void stomp_softmax_kernel(
    const float* __restrict__ costs,    // [B, K]
    float*       __restrict__ weights,  // [B, K]
    int K, float temperature)
{
    const int b = blockIdx.x;
    const int k = threadIdx.x;

    extern __shared__ float smem[];

    const float c = (k < K) ? costs[b * K + k] : 1e30f;

    // ── Step 1: Find min cost ──
    smem[k] = c;
    __syncthreads();
    block_reduce_min(smem, k);
    float min_c = smem[0];
    __syncthreads();

    float shifted = c - min_c;

    // ── Step 2: Compute mean of shifted costs ──
    smem[k] = (k < K) ? shifted : 0.0f;
    __syncthreads();
    block_reduce_sum(smem, k);
    float mean_shift = smem[0] / (float)K;
    __syncthreads();

    // ── Step 3: Compute variance → std → beta ──
    float diff = shifted - mean_shift;
    smem[k] = (k < K) ? diff * diff : 0.0f;
    __syncthreads();
    block_reduce_sum(smem, k);
    float std_shift = sqrtf(smem[0] / (float)K);
    float beta = fmaxf(std_shift, 1e-6f) * temperature;
    __syncthreads();

    // ── Step 4: Compute exp weights ──
    float w = (k < K) ? expf(-shifted / (beta + 1e-18f)) : 0.0f;
    smem[k] = w;
    __syncthreads();

    // ── Step 5: Sum and normalise ──
    block_reduce_sum(smem, k);
    float sum_w = smem[0];
    __syncthreads();

    if (k < K)
        weights[b * K + k] = w / (sum_w + 1e-30f);
}

// =========================================================================
// KERNEL 4: Weighted trajectory update (no atomics)
// =========================================================================
//
// Grid:  (B, T-2)         — one block per (batch, interior timestep)
// Block: next_pow2(K)     — one thread per sample
//
// Shared memory: [n_act * blockDim.x] floats (DOF-major layout for coalescing)
//
// For each DOF, each thread k loads weight[k] * noise[b,t,d,k],
// then a parallel reduction sums across K → delta[d].
// Thread 0 applies the update to the trajectory.

__global__ void stomp_update_kernel(
    float*       __restrict__ traj,     // [B, T, n_act] — updated in-place
    const float* __restrict__ noise,    // [B, T, n_act, K]
    const float* __restrict__ weights,  // [B, K]
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    int K, int T, int n_act, float step_size)
{
    const int b  = blockIdx.x;
    const int t  = blockIdx.y + 1;   // interior timestep: 1 .. T-2
    const int k  = threadIdx.x;

    extern __shared__ float smem[];  // [n_act * blockDim.x], DOF-major

    const float w = (k < K) ? weights[b * K + k] : 0.0f;
    const int block_K = blockDim.x;

    // Load weighted noise into shared memory (DOF-major for coalesced reduction)
    for (int d = 0; d < n_act; d++) {
        float n_val = (k < K)
            ? noise[NOISE_IDX(b, t, d, k, T, n_act, K)]
            : 0.0f;
        smem[d * block_K + k] = w * n_val;
    }
    __syncthreads();

    // Parallel reduction over K for each DOF simultaneously
    for (int s = block_K / 2; s > 0; s >>= 1) {
        if (k < s) {
            for (int d = 0; d < n_act; d++)
                smem[d * block_K + k] += smem[d * block_K + k + s];
        }
        __syncthreads();
    }

    // Thread 0 applies the update
    if (k == 0) {
        for (int d = 0; d < n_act; d++) {
            int idx = b * T * n_act + t * n_act + d;
            float val = traj[idx] + step_size * smem[d * block_K];
            traj[idx] = fmaxf(lower[d], fminf(upper[d], val));
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
    ffi::Buffer<ffi::DataType::F32>   L_inv_T,
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

    int T_in = T - 2;

    // ── Partition workspace buffer ──
    // Layout: noise[B*T*n_act*K] | costs[B*K] | weights[B*K]
    float* d_workspace = _workspace->typed_data();
    float* d_noise     = d_workspace;
    float* d_costs     = d_noise + B * T * n_act * K;
    float* d_weights   = d_costs + B * K;

    // ── Copy initial trajectories to output (updated in-place) ──
    float* d_traj = out_trajs->typed_data();
    cudaMemcpyAsync(d_traj, init_trajs.typed_data(),
                    B * T * n_act * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // ── Precompute block sizes (all power of 2) ──
    int T_block = next_pow2(T);    // for eval/final cost kernels
    int K_block = next_pow2(K);    // for softmax/update kernels

    // ── Noise kernel config ──
    int noise_block = T_in * n_act;
    int noise_smem  = (n_act * T_in + T_in * T_in) * (int)sizeof(float);

    // ── Eval cost kernel config ──
    int eval_smem = (T * n_act + T_block) * (int)sizeof(float);

    // ── Softmax kernel config ──
    int softmax_smem = K_block * (int)sizeof(float);

    // ── Update kernel config ──
    int update_smem = n_act * K_block * (int)sizeof(float);

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
    const float* d_L_inv_T    = L_inv_T.typed_data();

    uint64_t base_seed = (uint64_t)rng_seed;
    float w_coll = w_collision;

    // ═══════════════════════════════════════════════════════════════════════
    // Main STOMP loop — launch 4 kernels per iteration on the same stream
    // ═══════════════════════════════════════════════════════════════════════

    for (int iter = 0; iter < (int)n_iters; iter++) {

        // ── Kernel 1: Generate smooth noise ──
        stomp_smooth_noise_kernel<<<B * K, noise_block, noise_smem, stream>>>(
            d_noise, d_L_inv_T,
            B, K, T, n_act, T_in,
            noise_scale, base_seed, iter);

        // ── Kernel 2: Evaluate per-sample cost ──
        stomp_eval_cost_kernel<<<dim3(B, K), T_block, eval_smem, stream>>>(
            d_traj, d_noise, d_costs,
            d_twists, d_parent_tf, d_parent_idx, d_act_idx,
            d_mimic_mul, d_mimic_off, d_mimic_act, d_topo_inv,
            d_sphere_off, d_sphere_rad, d_pair_i, d_pair_j,
            d_world_s, d_world_c, d_world_b, d_world_h,
            d_lower, d_upper,
            T, n_joints, n_act, N, (int)S, P,
            Ms, Mc, Mb, Mh, K,
            w_smooth, w_acc, w_jerk, w_limits, w_coll, collision_margin);

        // ── Kernel 3: Softmax importance weights ──
        stomp_softmax_kernel<<<B, K_block, softmax_smem, stream>>>(
            d_costs, d_weights, K, temperature);

        // ── Kernel 4: Weighted trajectory update ──
        stomp_update_kernel<<<dim3(B, T - 2), K_block, update_smem, stream>>>(
            d_traj, d_noise, d_weights,
            d_lower, d_upper,
            K, T, n_act, step_size);

        // Scale collision penalty
        w_coll = fminf(w_coll * collision_penalty_scale, w_collision_max);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Final cost evaluation at w_collision_max
    // ═══════════════════════════════════════════════════════════════════════

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
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // L_inv_T
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
