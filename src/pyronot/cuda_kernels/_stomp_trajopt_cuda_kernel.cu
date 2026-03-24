/**
 * STOMP/MPPI TrajOpt CUDA kernel.
 *
 * Architecture:
 *   One CUDA **block** per trajectory (B blocks total).
 *   Within each block, K threads correspond to K noise samples:
 *     - Thread k evaluates sample k: loops over T timesteps, computing FK +
 *       collision + smoothness + limits cost for its perturbed trajectory.
 *     - All K threads barrier-sync; thread 0 computes softmax weights.
 *     - Each thread k replays its noise and atomically accumulates its
 *       weighted contribution into the shared delta buffer.
 *     - Thread 0 applies the update and scales the collision penalty.
 *
 * RNG replay:
 *   The per-sample noise is generated from a deterministic hash of
 *   (block_seed, iter, sample_k, timestep, dof).  This avoids storing
 *   K×T×DOF noise values by regenerating the same sequence in two passes.
 *
 * Shared memory layout per block:
 *   s_traj   [STOMP_MAX_T × STOMP_MAX_DOF]   current trajectory
 *   s_delta  [STOMP_MAX_T × STOMP_MAX_DOF]   weighted update accumulator
 *   s_costs  [STOMP_MAX_K]                   per-sample total costs
 *   s_weights[STOMP_MAX_K]                   importance weights
 *   s_robot  { FK param arrays }             loaded once per block
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
#define STOMP_MAX_T    64    // T=64 timesteps; halves local memory vs 128
#endif

#ifndef STOMP_MAX_DOF
#define STOMP_MAX_DOF  8     // Sufficient for 7-DOF arms (Panda, UR5, etc.)
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
#define STOMP_MAX_K    1024  // CUDA max threads/block; shared mem cost is only 2*K*4 bytes
#endif

// Maximum interior timesteps (T-2) for smooth noise shared memory.
// With STOMP_MAX_T=64, T_in=62: L_inv_T is 62²×4≈15 KB — fits within 48 KB
// shared memory alongside s_traj, s_delta, s_costs, s_weights.
#ifndef STOMP_SMOOTH_T_IN
#define STOMP_SMOOTH_T_IN  62
#endif

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
// Per-sample total cost  (smoothness + limits + collision over T timesteps)
// ---------------------------------------------------------------------------

static __device__ float stomp_sample_cost(
    const float q_k[STOMP_MAX_T][STOMP_MAX_DOF],   // perturbed trajectory
    int T, int n_act,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
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
    int n_joints, int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_coll, float collision_margin)
{
    float cost = 0.0f;

    // --- Smoothness (4th-order stencil: acc + jerk) ---
    const float ST[5] = {-1.f/12.f, 16.f/12.f, -30.f/12.f, 16.f/12.f, -1.f/12.f};
    for (int t = 0; t < T-4; t++) {
        for (int d = 0; d < n_act; d++) {
            float a = 0.0f;
            for (int k = 0; k < 5; k++) a += ST[k] * q_k[t+k][d];
            cost += w_smooth * w_acc * a * a;
        }
    }
    for (int t = 0; t < T-5; t++) {
        for (int d = 0; d < n_act; d++) {
            float a0 = 0.0f, a1 = 0.0f;
            for (int k = 0; k < 5; k++) {
                a0 += ST[k] * q_k[t+k][d];
                a1 += ST[k] * q_k[t+1+k][d];
            }
            float j = a1 - a0;
            cost += w_smooth * w_jerk * j * j;
        }
    }

    // FK workspace in local memory (thread-private)
    float T_world[STOMP_MAX_N * 7];

    // --- Per-timestep: limits + collision ---
    for (int t = 0; t < T; t++) {
        // Limits
        for (int d = 0; d < n_act; d++) {
            float viol = fmaxf(0.f, q_k[t][d] - upper[d])
                       + fmaxf(0.f, lower[d] - q_k[t][d]);
            cost += w_limits * viol * viol;
        }

        // Collision
        cost += w_coll * stomp_collision_cost_cfg(
            q_k[t], twists, parent_tf, parent_idx, act_idx,
            mimic_mul, mimic_off, mimic_act_idx, topo_inv,
            sphere_off, sphere_rad, pair_i, pair_j,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
            n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
            collision_margin, T_world);
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
    // Distinct Knuth multiplicative constants per dimension to avoid aliasing.
    // iter and d previously shared the same multiplier; when iter==d their XOR
    // contributions cancelled, producing identical noise across all iterations
    // for the diagonal entries — collapsing sample diversity.
    uint32_t state = (uint32_t)(base_seed
        ^ ((uint32_t)iter  * 2654435761u)
        ^ ((uint32_t)k     * 2246822519u)
        ^ ((uint32_t)t     * 3266489917u)
        ^ ((uint32_t)d     *    1664525u));  // different multiplier from iter
    if (state == 0u) state = 1u;
    return rng_normal(state) * noise_scale;
}

// ---------------------------------------------------------------------------
// STOMP kernel  —  blockDim.x = n_samples (K), gridDim.x = B
// ---------------------------------------------------------------------------

__global__ void stomp_trajopt_kernel(
    const float* __restrict__ init_trajs,
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
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const float* __restrict__ start,
    const float* __restrict__ goal,
    // L_inv_T: upper-triangular (T_in × T_in) matrix such that
    //   eps = L_inv_T @ z  →  eps ~ N(0, noise_scale² · M⁻¹)
    // where M = L L^T is the smoothness precision matrix.
    // NULL pointer → fall back to isotropic noise.
    const float* __restrict__ L_inv_T,
    float*       __restrict__ out_trajs,
    float*       __restrict__ out_costs,
    int T, int n_joints, int n_act, int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh, int n_iters,
    float noise_scale, float temperature, float step_size,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_collision, float w_collision_max,
    float collision_penalty_scale, float collision_margin,
    uint64_t base_seed)
{
    const int bid = blockIdx.x;    // trajectory index
    const int kid = threadIdx.x;   // sample index
    const int K   = blockDim.x;    // number of samples

    // ── Shared memory ─────────────────────────────────────────────────────
    __shared__ float s_traj   [STOMP_MAX_T * STOMP_MAX_DOF];
    __shared__ float s_delta  [STOMP_MAX_T * STOMP_MAX_DOF];
    __shared__ float s_costs  [STOMP_MAX_K];
    __shared__ float s_weights[STOMP_MAX_K];
    // Upper-triangular L_inv_T for smooth noise (zero-initialised; filled below).
    __shared__ float s_L_inv_T[STOMP_SMOOTH_T_IN * STOMP_SMOOTH_T_IN];

    // ── Load initial trajectory (thread 0 loads, all threads read) ────────
    if (kid == 0) {
        const float* src = init_trajs + bid * T * n_act;
        for (int i = 0; i < T * n_act; i++)
            s_traj[i] = src[i];
        // Pin endpoints
        for (int d = 0; d < n_act; d++) {
            s_traj[0*n_act+d]       = start[d];
            s_traj[(T-1)*n_act+d]   = goal[d];
        }
    }

    // ── Collaborative load of L_inv_T into shared memory ──────────────────
    // T_in = T - 2 interior timesteps.  Use smooth noise only when L_inv_T is
    // provided and T_in fits within the static shared-memory budget.
    const int T_in       = T - 2;
    const bool use_smooth = (L_inv_T != nullptr) && (T_in <= STOMP_SMOOTH_T_IN);
    if (use_smooth) {
        const int L_size = T_in * T_in;
        for (int i = kid; i < L_size; i += K)
            s_L_inv_T[i] = L_inv_T[i];
    }
    __syncthreads();

    // Per-block RNG seed incorporates block index
    const uint64_t block_seed = base_seed ^ ((uint64_t)bid * 6364136223846793005ULL);

    float w_coll = w_collision;

    // ── Main STOMP loop ───────────────────────────────────────────────────
    for (int iter = 0; iter < n_iters; iter++) {

        // ── Pass 1: build perturbed trajectory, evaluate cost, store noise ──
        // q_k holds the perturbed trajectory during cost eval, then is
        // converted to displacements (noise) for Pass 2 — eliminates the
        // expensive O(T²) L_inv_T replay that the old two-pass design needed.
        float q_k[STOMP_MAX_T][STOMP_MAX_DOF];

        // Pin endpoints
        for (int d = 0; d < n_act; d++) {
            q_k[0][d]     = s_traj[d];
            q_k[T-1][d]   = s_traj[(T-1)*n_act+d];
        }

        if (kid == 0) {
            // Null particle: current trajectory with no noise.
            for (int t = 1; t < T-1; t++)
                for (int d = 0; d < n_act; d++)
                    q_k[t][d] = s_traj[t*n_act+d];
        } else if (use_smooth) {
            // Smooth noise: eps = L_inv_T @ z  →  eps ~ N(0, noise_scale² · M⁻¹)
            float z_buf[STOMP_SMOOTH_T_IN];
            for (int d = 0; d < n_act; d++) {
                for (int i = 0; i < T_in; i++)
                    z_buf[i] = stomp_noise(block_seed, iter, kid, i, d, noise_scale);
                for (int i = 0; i < T_in; i++) {
                    float eps_i = 0.0f;
                    for (int j = i; j < T_in; j++)
                        eps_i += s_L_inv_T[i * T_in + j] * z_buf[j];
                    q_k[i + 1][d] = s_traj[(i + 1) * n_act + d] + eps_i;
                }
            }
        } else {
            // Isotropic noise
            for (int t = 1; t < T-1; t++)
                for (int d = 0; d < n_act; d++)
                    q_k[t][d] = s_traj[t*n_act+d]
                              + stomp_noise(block_seed, iter, kid, t, d, noise_scale);
        }

        // Evaluate full cost
        float c = stomp_sample_cost(
            q_k, T, n_act, lower, upper,
            twists, parent_tf, parent_idx, act_idx,
            mimic_mul, mimic_off, mimic_act_idx, topo_inv,
            sphere_off, sphere_rad, pair_i, pair_j,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
            n_joints, N, S, P, Ms, Mc, Mb, Mh,
            w_smooth, w_acc, w_jerk, w_limits, w_coll, collision_margin);

        s_costs[kid] = c;

        // Convert q_k from absolute positions to displacements (noise).
        // After this, q_k[t][d] = perturbation applied at (t, d).
        // Null particle (kid==0) gets zeros since q_k == s_traj.
        for (int t = 1; t < T-1; t++)
            for (int d = 0; d < n_act; d++)
                q_k[t][d] -= s_traj[t*n_act+d];

        __syncthreads();

        // ── Thread 0: compute softmax importance weights ───────────────────
        if (kid == 0) {
            float min_c = s_costs[0];
            for (int k = 1; k < K; k++)
                if (s_costs[k] < min_c) min_c = s_costs[k];

            float mean_shift = 0.0f;
            for (int k = 0; k < K; k++) mean_shift += (s_costs[k] - min_c);
            mean_shift /= (float)K;
            float var_shift = 0.0f;
            for (int k = 0; k < K; k++) {
                float d = (s_costs[k] - min_c) - mean_shift;
                var_shift += d * d;
            }
            float std_shift = sqrtf(var_shift / (float)K);
            float beta = fmaxf(std_shift, 1e-6f) * temperature;

            float sum_w = 0.0f;
            for (int k = 0; k < K; k++) {
                float w = expf(-(s_costs[k] - min_c) / (beta + 1e-18f));
                s_weights[k] = w;
                sum_w += w;
            }
            float inv_sum = 1.0f / (sum_w + 1e-30f);
            for (int k = 0; k < K; k++)
                s_weights[k] *= inv_sum;
        }

        // Zero delta accumulator — all K threads participate (was thread-0 only)
        for (int i = kid; i < T * n_act; i += K)
            s_delta[i] = 0.0f;
        __syncthreads();

        // ── Pass 2: accumulate weighted noise from stored displacements ────
        // No L_inv_T replay — just read the displacement already in q_k.
        if (kid != 0) {
            float wk = s_weights[kid];
            for (int t = 1; t < T-1; t++)
                for (int d = 0; d < n_act; d++)
                    atomicAdd(&s_delta[t*n_act+d], wk * q_k[t][d]);
        }
        __syncthreads();

        // ── Thread 0: apply update, pin endpoints, scale collision ─────────
        if (kid == 0) {
            for (int t = 1; t < T-1; t++) {
                for (int d = 0; d < n_act; d++) {
                    s_traj[t*n_act+d] += step_size * s_delta[t*n_act+d];
                    s_traj[t*n_act+d] = fmaxf(lower[d], fminf(upper[d], s_traj[t*n_act+d]));
                }
            }
            for (int d = 0; d < n_act; d++) {
                s_traj[0*n_act+d]     = start[d];
                s_traj[(T-1)*n_act+d] = goal[d];
            }
            w_coll = fminf(w_coll * collision_penalty_scale, w_collision_max);
        }
        __syncthreads();
    }  // end main STOMP loop

    // ── Store final trajectory and evaluate final cost ─────────────────────
    float* dst = out_trajs + bid * T * n_act;
    if (kid == 0) {
        for (int i = 0; i < T * n_act; i++)
            dst[i] = s_traj[i];
    }
    __syncthreads();

    // Final cost: thread kid evaluates one sample at w_collision_max
    // (no noise: sample = current trajectory)
    if (kid == 0) {
        float q_final[STOMP_MAX_T][STOMP_MAX_DOF];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < n_act; d++)
                q_final[t][d] = s_traj[t*n_act+d];

        float final_c = stomp_sample_cost(
            q_final, T, n_act, lower, upper,
            twists, parent_tf, parent_idx, act_idx,
            mimic_mul, mimic_off, mimic_act_idx, topo_inv,
            sphere_off, sphere_rad, pair_i, pair_j,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
            n_joints, N, S, P, Ms, Mc, Mb, Mh,
            w_smooth, w_acc, w_jerk, w_limits, w_collision_max, collision_margin);

        out_costs[bid] = final_c;
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
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
    ffi::Buffer<ffi::DataType::F32>   L_inv_T,   // (T-2)×(T-2) upper-triangular smooth-noise matrix
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
    // Extract dimensions
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

    dim3 grid(B);
    dim3 block(K);

    stomp_trajopt_kernel<<<grid, block, 0, stream>>>(
        init_trajs.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        sphere_off.typed_data(),
        sphere_rad.typed_data(),
        pair_i.typed_data(),
        pair_j.typed_data(),
        world_spheres.typed_data(),
        world_capsules.typed_data(),
        world_boxes.typed_data(),
        world_halfspaces.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        start.typed_data(),
        goal.typed_data(),
        L_inv_T.typed_data(),
        out_trajs->typed_data(),
        out_costs->typed_data(),
        T, n_joints, n_act, N, (int)S, P,
        Ms, Mc, Mb, Mh, (int)n_iters,
        noise_scale, temperature, step_size,
        w_smooth, w_acc, w_jerk,
        w_limits, w_collision, w_collision_max,
        collision_penalty_scale, collision_margin,
        (uint64_t)rng_seed);

    return ffi::Error::Success();
}

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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // workspace (unused, keeps shape constant)
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
