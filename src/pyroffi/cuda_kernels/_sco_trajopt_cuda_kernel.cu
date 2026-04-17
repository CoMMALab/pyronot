/**
 * SCO TrajOpt CUDA kernel — block-parallel redesign.
 *
 * Architecture:
 *   One CUDA **block** per trajectory (B blocks total).
 *   Within each block, threads cooperate over T timesteps:
 *     - FK + finite-difference Jacobians: 1 thread per timestep (parallel).
 *     - Inner L-BFGS cost/gradient: 1 thread per timestep + block reduction.
 *     - L-BFGS two-loop: parallel dot-products via block reduction.
 *     - Line search: parallel cost evaluation for 5 alpha candidates.
 *
 * Shared memory holds the trajectory (s_traj), linearisation point (s_qk),
 * collision distances (s_dk), and reduction scratch.  Robot parameters are
 * in static shared memory.  All other buffers live in per-block global
 * workspace.
 *
 * Build with:
 *   bash src/pyroffi/cuda_kernels/build_sco_trajopt_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "_collision_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA graph cache for the SCO TrajOpt kernel
// ---------------------------------------------------------------------------
// Amortises CPU-side kernel-launch overhead (driver API call, argument
// marshalling) to a single cudaGraphLaunch per FFI call.  The graph is
// recaptured only when the problem shape changes.

struct ScoTrajoptGraphCache {
    cudaGraphExec_t  exec           = nullptr;
    cudaGraphNode_t  kernel_node    = nullptr;
    cudaGraph_t      graph          = nullptr;
    cudaStream_t     capture_stream = nullptr;

    void*        func_ptr   = nullptr;
    dim3         grid_dim   = {1, 1, 1};
    dim3         block_dim  = {1, 1, 1};
    unsigned int shared_mem = 0;

    // Shape fingerprint — invalidated whenever any dimension changes.
    int fp_B = -1, fp_T = -1, fp_n_act = -1, fp_n_joints = -1;
    int fp_N = -1, fp_S = -1, fp_P = -1;
    int fp_Ms = -1, fp_Mc = -1, fp_Mb = -1, fp_Mh = -1;

    bool shape_matches(int B, int T, int n_act, int n_joints,
                       int N, int S, int P,
                       int Ms, int Mc, int Mb, int Mh) const noexcept {
        return B == fp_B && T == fp_T && n_act == fp_n_act &&
               n_joints == fp_n_joints && N == fp_N && S == fp_S &&
               P == fp_P && Ms == fp_Ms && Mc == fp_Mc &&
               Mb == fp_Mb && Mh == fp_Mh;
    }

    cudaError_t ensure_capture_stream() noexcept {
        if (capture_stream) return cudaSuccess;
        return cudaStreamCreateWithFlags(&capture_stream,
                                         cudaStreamNonBlocking);
    }

    void invalidate() noexcept {
        if (exec)  { cudaGraphExecDestroy(exec);  exec  = nullptr; }
        if (graph) { cudaGraphDestroy(graph);      graph = nullptr; }
        kernel_node = nullptr;
        func_ptr    = nullptr;
        fp_B = fp_T = fp_n_act = fp_n_joints = -1;
        fp_N = fp_S = fp_P = -1;
        fp_Ms = fp_Mc = fp_Mb = fp_Mh = -1;
    }

    cudaError_t finalize_capture(int B, int T, int n_act, int n_joints,
                                  int N, int S, int P,
                                  int Ms, int Mc, int Mb, int Mh) noexcept {
        size_t n_nodes = 1;
        cudaError_t e = cudaGraphGetNodes(graph, &kernel_node, &n_nodes);
        if (e != cudaSuccess) return e;
        if (n_nodes == 0)     return cudaErrorUnknown;

        cudaKernelNodeParams kp = {};
        e = cudaGraphKernelNodeGetParams(kernel_node, &kp);
        if (e != cudaSuccess) return e;
        func_ptr   = kp.func;
        grid_dim   = kp.gridDim;
        block_dim  = kp.blockDim;
        shared_mem = kp.sharedMemBytes;

        e = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        if (e != cudaSuccess) return e;

        fp_B = B; fp_T = T; fp_n_act = n_act; fp_n_joints = n_joints;
        fp_N = N; fp_S = S; fp_P = P;
        fp_Ms = Ms; fp_Mc = Mc; fp_Mb = Mb; fp_Mh = Mh;
        return cudaSuccess;
    }
};

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits
// ---------------------------------------------------------------------------

#ifndef SCO_MAX_T
#define SCO_MAX_T     128
#endif

#ifndef SCO_MAX_DOF
#define SCO_MAX_DOF   MAX_ACT   // 16, from _ik_cuda_helpers.cuh
#endif

#ifndef SCO_MAX_N
#define SCO_MAX_N     MAX_JOINTS  // 64
#endif

#ifndef SCO_MAX_S
#define SCO_MAX_S     8
#endif

#define SCO_MAX_G     5    // 1 self + 4 world types

#ifndef SCO_MAX_M
#define SCO_MAX_M     8
#endif

#ifndef SCO_MAX_PAIRS
#define SCO_MAX_PAIRS 256
#endif

// ---------------------------------------------------------------------------
// Warp / block reduction
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float block_reduce_sum(float val, float* smem, int tid, int bdim) {
    int lane    = tid & 31;
    int warp_id = tid >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    int n_warps = bdim >> 5;
    float ws = (tid < n_warps) ? smem[tid] : 0.0f;
    ws = warp_reduce_sum(ws);

    if (tid == 0) smem[0] = ws;
    __syncthreads();
    return smem[0];
}

// ---------------------------------------------------------------------------
// Collision distance primitives (shared)
// ---------------------------------------------------------------------------

#define sco_sphere_sphere_dist sphere_sphere_dist
#define sco_sphere_capsule_dist sphere_capsule_dist
#define sco_sphere_box_dist sphere_box_dist
#define sco_sphere_halfspace_dist sphere_halfspace_dist
#define sco_apply_se3 apply_se3_point
#define sco_colldist_from_sdf colldist_from_sdf

// ---------------------------------------------------------------------------
// Incremental smooth-min accumulator
// ---------------------------------------------------------------------------

struct SmoothMinAcc {
    float max_val, sum_exp;
    __device__ void init() { max_val = -1e30f; sum_exp = 0.0f; }
    __device__ void update(float d, float tau) {
        float v = -d / tau;
        if (v > max_val) { sum_exp *= expf(max_val - v); max_val = v; }
        sum_exp += expf(v - max_val);
    }
    __device__ float finalize(float tau) const {
        return (sum_exp <= 0.0f) ? 1e10f : -tau * (logf(sum_exp) + max_val);
    }
};

// ---------------------------------------------------------------------------
// Per-configuration collision distances (G=5 smooth-min groups)
// ---------------------------------------------------------------------------

__device__ void sco_compute_coll_dists(
    const float* __restrict__ cfg,
    const float* __restrict__ s_twists,
    const float* __restrict__ s_parent_tf,
    const int*   __restrict__ s_parent_idx,
    const int*   __restrict__ s_act_idx,
    const float* __restrict__ s_mimic_mul,
    const float* __restrict__ s_mimic_off,
    const int*   __restrict__ s_mimic_act_idx,
    const int*   __restrict__ s_topo_inv,
    const float* __restrict__ s_sphere_off,
    const float* __restrict__ s_sphere_rad,
    const int*   __restrict__ s_pair_i,
    const int*   __restrict__ s_pair_j,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    int n_joints, int n_act, int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    float temperature,
    float* __restrict__ dists,
    float* __restrict__ T_world)
{
    fk_single(cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
              s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
              T_world, n_joints, n_act);

    SmoothMinAcc acc[SCO_MAX_G];
    for (int g = 0; g < SCO_MAX_G; g++) acc[g].init();

    // Group 0: self-collision — hard min over spheres per link pair,
    // then smooth-min over link pairs (matches JAX's compute_self_collision_distance)
    for (int p = 0; p < P; p++) {
        int li = s_pair_i[p], lj = s_pair_j[p];
        float min_d = 1e10f;
        for (int si = 0; si < S; si++) {
            float ri = s_sphere_rad[li*S+si];
            if (ri < 0.0f) continue;
            float lci[3] = {s_sphere_off[(li*S+si)*3],
                            s_sphere_off[(li*S+si)*3+1],
                            s_sphere_off[(li*S+si)*3+2]};
            float ci[3]; sco_apply_se3(T_world + li*7, lci, ci);
            for (int sj = 0; sj < S; sj++) {
                float rj = s_sphere_rad[lj*S+sj];
                if (rj < 0.0f) continue;
                float lcj[3] = {s_sphere_off[(lj*S+sj)*3],
                                s_sphere_off[(lj*S+sj)*3+1],
                                s_sphere_off[(lj*S+sj)*3+2]};
                float cj[3]; sco_apply_se3(T_world + lj*7, lcj, cj);
                float d = sco_sphere_sphere_dist(ci[0],ci[1],ci[2],ri,
                                                  cj[0],cj[1],cj[2],rj);
                if (d < min_d) min_d = d;
            }
        }
        if (min_d < 1e9f) acc[0].update(min_d, temperature);
    }

    // Groups 1-4: world obstacles — hard min over spheres per (link, obstacle)
    // pair, then smooth-min over all pairs (matches JAX's compute_world_collision_distance)
    for (int j = 0; j < N; j++) {
        for (int m = 0; m < Ms; m++) {
            float min_d = 1e10f;
            for (int s = 0; s < S; s++) {
                float r = s_sphere_rad[j*S+s];
                if (r < 0.0f) continue;
                float lc[3] = {s_sphere_off[(j*S+s)*3],
                               s_sphere_off[(j*S+s)*3+1],
                               s_sphere_off[(j*S+s)*3+2]};
                float wc[3]; sco_apply_se3(T_world + j*7, lc, wc);
                float d = sco_sphere_sphere_dist(
                    wc[0],wc[1],wc[2],r,
                    world_spheres[m*4],world_spheres[m*4+1],
                    world_spheres[m*4+2],world_spheres[m*4+3]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) acc[1].update(min_d, temperature);
        }
        for (int m = 0; m < Mc; m++) {
            float min_d = 1e10f;
            const float* cap = world_capsules + m*7;
            for (int s = 0; s < S; s++) {
                float r = s_sphere_rad[j*S+s];
                if (r < 0.0f) continue;
                float lc[3] = {s_sphere_off[(j*S+s)*3],
                               s_sphere_off[(j*S+s)*3+1],
                               s_sphere_off[(j*S+s)*3+2]};
                float wc[3]; sco_apply_se3(T_world + j*7, lc, wc);
                float d = sco_sphere_capsule_dist(
                    wc[0],wc[1],wc[2],r,
                    cap[0],cap[1],cap[2],cap[3],cap[4],cap[5],cap[6]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) acc[2].update(min_d, temperature);
        }
        for (int m = 0; m < Mb; m++) {
            float min_d = 1e10f;
            const float* bx = world_boxes + m*15;
            for (int s = 0; s < S; s++) {
                float r = s_sphere_rad[j*S+s];
                if (r < 0.0f) continue;
                float lc[3] = {s_sphere_off[(j*S+s)*3],
                               s_sphere_off[(j*S+s)*3+1],
                               s_sphere_off[(j*S+s)*3+2]};
                float wc[3]; sco_apply_se3(T_world + j*7, lc, wc);
                float d = sco_sphere_box_dist(
                    wc[0],wc[1],wc[2],r,
                    bx[0],bx[1],bx[2], bx[3],bx[4],bx[5],
                    bx[6],bx[7],bx[8], bx[9],bx[10],bx[11],
                    bx[12],bx[13],bx[14]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) acc[3].update(min_d, temperature);
        }
        for (int m = 0; m < Mh; m++) {
            float min_d = 1e10f;
            const float* hs = world_halfspaces + m*6;
            for (int s = 0; s < S; s++) {
                float r = s_sphere_rad[j*S+s];
                if (r < 0.0f) continue;
                float lc[3] = {s_sphere_off[(j*S+s)*3],
                               s_sphere_off[(j*S+s)*3+1],
                               s_sphere_off[(j*S+s)*3+2]};
                float wc[3]; sco_apply_se3(T_world + j*7, lc, wc);
                float d = sco_sphere_halfspace_dist(
                    wc[0],wc[1],wc[2],r,
                    hs[0],hs[1],hs[2],hs[3],hs[4],hs[5]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) acc[4].update(min_d, temperature);
        }
    }

    for (int g = 0; g < SCO_MAX_G; g++) dists[g] = acc[g].finalize(temperature);
}

// ---------------------------------------------------------------------------
// Per-timestep inner cost + gradient  (called by one thread per timestep)
// ---------------------------------------------------------------------------

static __device__ void sco_inner_costgrad_timestep(
    int t, int T, int n_act,
    const float* __restrict__ s_traj,   // shared [T*n_act]
    const float* __restrict__ s_qk,     // shared [T*n_act]
    const float* __restrict__ s_dk,     // shared [T*G]
    const float* __restrict__ J_k,      // global [T*G*n_act]
    const float* __restrict__ lo,       // shared [n_act]
    const float* __restrict__ hi,       // shared [n_act]
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_trust, float w_coll,
    float collision_margin,
    float* __restrict__ out_cost,
    float* __restrict__ grad)           // global, writes [t*n_act .. (t+1)*n_act-1]
{
    const float ST[5] = {-1.f/12.f, 16.f/12.f, -30.f/12.f, 16.f/12.f, -1.f/12.f};
    int n_acc  = (T >= 5) ? T - 4 : 0;
    int n_jerk = (T >= 6) ? T - 5 : 0;
    float cost = 0.0f;

    // Zero grad for this timestep
    for (int d = 0; d < n_act; d++) grad[t*n_act+d] = 0.0f;

    // ---- Smoothness cost (owned windows) ----
    if (t < n_acc) {
        for (int d = 0; d < n_act; d++) {
            float a = 0.0f;
            for (int k = 0; k < 5; k++) a += ST[k] * s_traj[(t+k)*n_act+d];
            cost += w_smooth * w_acc * a * a;
        }
    }
    if (t < n_jerk) {
        for (int d = 0; d < n_act; d++) {
            float a0 = 0.0f, a1 = 0.0f;
            for (int k = 0; k < 5; k++) {
                a0 += ST[k] * s_traj[(t+k)*n_act+d];
                a1 += ST[k] * s_traj[(t+1+k)*n_act+d];
            }
            float j = a1 - a0;
            cost += w_smooth * w_jerk * j * j;
        }
    }

    // ---- Smoothness gradient (recompute acc on-the-fly) ----
    {
        int s = t;
        for (int d = 0; d < n_act; d++) {
            float g = 0.0f;
            int tlo = (s >= 4) ? s - 4 : 0;
            int thi = (s < n_acc) ? s : n_acc - 1;
            for (int tt = tlo; tt <= thi; tt++) {
                float a = 0.0f;
                for (int k = 0; k < 5; k++) a += ST[k] * s_traj[(tt+k)*n_act+d];
                g += 2.0f * w_acc * a * ST[s - tt];
            }
            int jlo = (s >= 5) ? s - 5 : 0;
            int jhi = (s < n_jerk) ? s : n_jerk - 1;
            for (int tt = jlo; tt <= jhi; tt++) {
                float a0 = 0.0f, a1 = 0.0f;
                for (int k = 0; k < 5; k++) {
                    a0 += ST[k] * s_traj[(tt+k)*n_act+d];
                    a1 += ST[k] * s_traj[(tt+1+k)*n_act+d];
                }
                float jk = a1 - a0;
                int k1 = s - tt - 1, k2 = s - tt;
                float c1 = (k1 >= 0 && k1 <= 4) ? ST[k1] : 0.0f;
                float c2 = (k2 >= 0 && k2 <= 4) ? ST[k2] : 0.0f;
                g += 2.0f * w_jerk * jk * (c1 - c2);
            }
            grad[s*n_act+d] += w_smooth * g;
        }
    }

    // ---- Limits ----
    for (int d = 0; d < n_act; d++) {
        float qv = s_traj[t*n_act+d];
        float viol = fmaxf(0.0f, qv - hi[d]) + fmaxf(0.0f, lo[d] - qv);
        cost += w_limits * viol * viol;
        float sign = (qv > hi[d]) ? 1.0f : (qv < lo[d]) ? -1.0f : 0.0f;
        grad[t*n_act+d] += 2.0f * w_limits * viol * sign;
    }

    // ---- Linearised collision ----
    for (int g = 0; g < SCO_MAX_G; g++) {
        float d_lin = s_dk[t*SCO_MAX_G+g];
        for (int d = 0; d < n_act; d++)
            d_lin += J_k[(t*SCO_MAX_G+g)*n_act+d] *
                     (s_traj[t*n_act+d] - s_qk[t*n_act+d]);
        float viol = fmaxf(0.0f, collision_margin - d_lin);
        cost += w_coll * viol * viol;
        float gc = -2.0f * w_coll * viol;
        for (int d = 0; d < n_act; d++)
            grad[t*n_act+d] += gc * J_k[(t*SCO_MAX_G+g)*n_act+d];
    }

    // ---- Trust region ----
    for (int d = 0; d < n_act; d++) {
        float delta = s_traj[t*n_act+d] - s_qk[t*n_act+d];
        cost += w_trust * delta * delta;
        grad[t*n_act+d] += 2.0f * w_trust * delta;
    }

    // ---- Pin endpoints ----
    if (t == 0 || t == T - 1)
        for (int d = 0; d < n_act; d++) grad[t*n_act+d] = 0.0f;

    *out_cost = cost;
}

// ---------------------------------------------------------------------------
// Per-timestep trial cost (no gradient — used in line search)
// Evaluates cost at x = s_traj + alpha * dir
// ---------------------------------------------------------------------------

static __device__ float sco_trial_cost_timestep(
    int t, float alpha, int T, int n_act,
    const float* __restrict__ s_traj,
    const float* __restrict__ s_qk,
    const float* __restrict__ s_dk,
    const float* __restrict__ J_k,
    const float* __restrict__ dir,
    const float* __restrict__ lo,
    const float* __restrict__ hi,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_trust, float w_coll,
    float collision_margin)
{
    const float ST[5] = {-1.f/12.f, 16.f/12.f, -30.f/12.f, 16.f/12.f, -1.f/12.f};
    int n_acc  = (T >= 5) ? T - 4 : 0;
    int n_jerk = (T >= 6) ? T - 5 : 0;
    float cost = 0.0f;

    // Macro: trial value at (s, d)
    #define XV(s, d) (s_traj[(s)*n_act+(d)] + alpha * dir[(s)*n_act+(d)])

    if (t < n_acc) {
        for (int d = 0; d < n_act; d++) {
            float a = 0.0f;
            for (int k = 0; k < 5; k++) a += ST[k] * XV(t+k, d);
            cost += w_smooth * w_acc * a * a;
        }
    }
    if (t < n_jerk) {
        for (int d = 0; d < n_act; d++) {
            float a0 = 0.0f, a1 = 0.0f;
            for (int k = 0; k < 5; k++) {
                a0 += ST[k] * XV(t+k, d);
                a1 += ST[k] * XV(t+1+k, d);
            }
            float j = a1 - a0; cost += w_smooth * w_jerk * j * j;
        }
    }

    for (int d = 0; d < n_act; d++) {
        float qv = XV(t, d);
        float viol = fmaxf(0.0f, qv - hi[d]) + fmaxf(0.0f, lo[d] - qv);
        cost += w_limits * viol * viol;
    }

    for (int g = 0; g < SCO_MAX_G; g++) {
        float d_lin = s_dk[t*SCO_MAX_G+g];
        for (int d = 0; d < n_act; d++)
            d_lin += J_k[(t*SCO_MAX_G+g)*n_act+d] *
                     (XV(t, d) - s_qk[t*n_act+d]);
        float viol = fmaxf(0.0f, collision_margin - d_lin);
        cost += w_coll * viol * viol;
    }

    for (int d = 0; d < n_act; d++) {
        float delta = XV(t, d) - s_qk[t*n_act+d];
        cost += w_trust * delta * delta;
    }
    #undef XV
    return cost;
}

// ---------------------------------------------------------------------------
// Cooperative L-BFGS two-loop recursion
//   Each thread handles elements [tid*n_act .. (tid+1)*n_act-1] of vectors
//   of length T*n_act stored in global memory.
// ---------------------------------------------------------------------------

static __device__ void sco_lbfgs_two_loop_coop(
    const float* __restrict__ g_vec,   // [T*n_act] gradient
    const float* __restrict__ s_buf,   // [m_max * T*n_act]
    const float* __restrict__ y_buf,   // [m_max * T*n_act]
    const float* __restrict__ rho_buf, // [m_max]
    float*       __restrict__ ah,      // [m_max] alpha scratch
    float*       __restrict__ p,       // [T*n_act] output direction
    int T, int n_act, int m_max, int m_used, int newest,
    int tid, int bdim, float* smem_r)
{
    int n = T * n_act;
    int i0 = tid * n_act;
    int i1 = (tid < T) ? i0 + n_act : i0;

    // p = -g
    for (int i = i0; i < i1; i++) p[i] = -g_vec[i];

    if (m_used == 0) {
        // Normalised steepest descent (matches JAX: -g / ||g||)
        float loc_gg = 0.0f;
        for (int i = i0; i < i1; i++) loc_gg += g_vec[i] * g_vec[i];
        float gg = block_reduce_sum(loc_gg, smem_r, tid, bdim);
        float inv_norm = 1.0f / (sqrtf(gg) + 1e-18f);
        for (int i = i0; i < i1; i++) p[i] *= inv_norm;
        return;
    }

    // First loop: newest → oldest
    for (int k = 0; k < m_used; k++) {
        int idx = (newest - k + m_max) % m_max;
        float loc = 0.0f;
        for (int i = i0; i < i1; i++)
            loc += s_buf[idx*n+i] * p[i];
        float alpha_k = block_reduce_sum(loc, smem_r, tid, bdim) * rho_buf[idx];
        if (tid == 0) ah[k] = alpha_k;
        // alpha_k is the same for all threads (from reduction)
        for (int i = i0; i < i1; i++)
            p[i] -= alpha_k * y_buf[idx*n+i];
    }

    // Gamma scaling (Shanno-Kettler H0)
    {
        float loc_sy = 0.0f, loc_yy = 0.0f;
        for (int i = i0; i < i1; i++) {
            loc_sy += s_buf[newest*n+i] * y_buf[newest*n+i];
            loc_yy += y_buf[newest*n+i] * y_buf[newest*n+i];
        }
        float sy = block_reduce_sum(loc_sy, smem_r, tid, bdim);
        float yy = block_reduce_sum(loc_yy, smem_r, tid, bdim);
        float gamma = sy / (yy + 1e-18f);   // unclamped, matches JAX
        for (int i = i0; i < i1; i++) p[i] *= gamma;
    }

    // Second loop: oldest → newest
    for (int k = m_used - 1; k >= 0; k--) {
        int idx = (newest - k + m_max) % m_max;
        float loc = 0.0f;
        for (int i = i0; i < i1; i++)
            loc += y_buf[idx*n+i] * p[i];
        float beta = block_reduce_sum(loc, smem_r, tid, bdim) * rho_buf[idx];
        float coeff = ah[k] - beta;
        for (int i = i0; i < i1; i++)
            p[i] += s_buf[idx*n+i] * coeff;
    }
}

// ---------------------------------------------------------------------------
// Main kernel — one block per trajectory
// ---------------------------------------------------------------------------

__global__
void sco_trajopt_kernel(
    const float* __restrict__ init_trajs,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const float* __restrict__ sphere_offsets,
    const float* __restrict__ sphere_radii,
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
    float*       __restrict__ out_trajs,
    float*       __restrict__ out_costs,
    float*       __restrict__ workspace,
    int workspace_stride,
    int B, int T, int n_joints, int n_act,
    int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    int n_outer_iters, int n_inner_iters, int m_lbfgs,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_trust,
    float w_collision, float w_collision_max, float penalty_scale,
    float collision_margin, float smooth_min_temperature, float fd_eps)
{
    // ---- Static shared memory: robot params ----
    __shared__ float s_twists       [MAX_JOINTS * 6];
    __shared__ float s_parent_tf    [MAX_JOINTS * 7];
    __shared__ int   s_parent_idx   [MAX_JOINTS];
    __shared__ int   s_act_idx      [MAX_JOINTS];
    __shared__ float s_mimic_mul    [MAX_JOINTS];
    __shared__ float s_mimic_off    [MAX_JOINTS];
    __shared__ int   s_mimic_act_idx[MAX_JOINTS];
    __shared__ int   s_topo_inv     [MAX_JOINTS];
    __shared__ float s_sphere_off   [SCO_MAX_N * SCO_MAX_S * 3];
    __shared__ float s_sphere_rad   [SCO_MAX_N * SCO_MAX_S];
    __shared__ int   s_pair_i       [SCO_MAX_PAIRS];
    __shared__ int   s_pair_j       [SCO_MAX_PAIRS];
    __shared__ float s_lower        [SCO_MAX_DOF];
    __shared__ float s_upper        [SCO_MAX_DOF];
    __shared__ float s_start        [SCO_MAX_DOF];
    __shared__ float s_goal         [SCO_MAX_DOF];

    // Cooperative load
    for (int i = threadIdx.x; i < n_joints*6; i += blockDim.x) s_twists[i]    = twists[i];
    for (int i = threadIdx.x; i < n_joints*7; i += blockDim.x) s_parent_tf[i] = parent_tf[i];
    for (int i = threadIdx.x; i < n_joints;   i += blockDim.x) {
        s_parent_idx[i]    = parent_idx[i];
        s_act_idx[i]       = act_idx[i];
        s_mimic_mul[i]     = mimic_mul[i];
        s_mimic_off[i]     = mimic_off[i];
        s_mimic_act_idx[i] = mimic_act_idx[i];
        s_topo_inv[i]      = topo_inv[i];
    }
    int ns3 = N*S*3, ns = N*S;
    for (int i = threadIdx.x; i < ns3; i += blockDim.x) s_sphere_off[i] = sphere_offsets[i];
    for (int i = threadIdx.x; i < ns;  i += blockDim.x) s_sphere_rad[i] = sphere_radii[i];
    for (int i = threadIdx.x; i < P;   i += blockDim.x) {
        s_pair_i[i] = pair_i[i];
        s_pair_j[i] = pair_j[i];
    }
    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i] = lower[i]; s_upper[i] = upper[i];
        s_start[i] = start[i]; s_goal[i]  = goal[i];
    }
    __syncthreads();

    const int b    = blockIdx.x;
    if (b >= B) return;
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;
    const int n    = T * n_act;

    // ---- Dynamic shared memory ----
    extern __shared__ float dyn[];
    float* s_traj  = dyn;                       // [T * n_act]
    float* s_qk    = s_traj + n;                // [T * n_act]
    float* s_dk    = s_qk   + n;                // [T * SCO_MAX_G]
    float* smem_r  = s_dk   + T * SCO_MAX_G;    // [bdim] reduction scratch

    // ---- Per-block global workspace ----
    float* base      = workspace + (size_t)b * workspace_stride;
    float* grad      = base;
    float* dir       = grad   + n;
    float* best_x    = dir    + n;
    float* g_prev    = best_x + n;
    float* J_k       = g_prev + n;
    float* s_lbfgs   = J_k    + T * SCO_MAX_G * n_act;
    float* y_lbfgs   = s_lbfgs + m_lbfgs * n;
    float* rho_buf   = y_lbfgs + m_lbfgs * n;
    float* alpha_hist= rho_buf + SCO_MAX_M;
    float* T_world_pool = alpha_hist + SCO_MAX_M;

    // ---- Load initial trajectory into shared memory ----
    for (int i = tid; i < n; i += bdim)
        s_traj[i] = init_trajs[b * n + i];
    if (tid < n_act) {
        s_traj[tid]               = s_start[tid];
        s_traj[(T-1)*n_act + tid] = s_goal[tid];
    }
    __syncthreads();

    float w_coll = w_collision;

    // ==== OUTER SCO LOOP ====
    for (int outer = 0; outer < n_outer_iters; outer++) {

        // ---- Copy traj → q_k ----
        for (int i = tid; i < n; i += bdim) s_qk[i] = s_traj[i];
        __syncthreads();

        // ---- FK + FD Jacobians (one thread per timestep) ----
        if (tid < T) {
            float* my_Tw = T_world_pool + tid * n_joints * 7;
            float d_base[SCO_MAX_G];

            sco_compute_coll_dists(
                s_traj + tid * n_act,
                s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                s_sphere_off, s_sphere_rad, s_pair_i, s_pair_j,
                world_spheres, world_capsules, world_boxes, world_halfspaces,
                n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
                smooth_min_temperature, d_base, my_Tw);

            for (int g = 0; g < SCO_MAX_G; g++)
                s_dk[tid * SCO_MAX_G + g] = d_base[g];

            // Central finite-difference Jacobians
            float q_pert[SCO_MAX_DOF];
            for (int a = 0; a < n_act; a++)
                q_pert[a] = s_traj[tid * n_act + a];

            for (int d = 0; d < n_act; d++) {
                float orig = q_pert[d];
                float dp[SCO_MAX_G], dm[SCO_MAX_G];

                q_pert[d] = orig + fd_eps;
                sco_compute_coll_dists(
                    q_pert,
                    s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                    s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                    s_sphere_off, s_sphere_rad, s_pair_i, s_pair_j,
                    world_spheres, world_capsules, world_boxes, world_halfspaces,
                    n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
                    smooth_min_temperature, dp, my_Tw);

                q_pert[d] = orig - fd_eps;
                sco_compute_coll_dists(
                    q_pert,
                    s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                    s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                    s_sphere_off, s_sphere_rad, s_pair_i, s_pair_j,
                    world_spheres, world_capsules, world_boxes, world_halfspaces,
                    n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
                    smooth_min_temperature, dm, my_Tw);

                q_pert[d] = orig;
                float inv2e = 1.0f / (2.0f * fd_eps);
                for (int g = 0; g < SCO_MAX_G; g++)
                    J_k[(tid*SCO_MAX_G+g)*n_act + d] = (dp[g] - dm[g]) * inv2e;
            }
        }
        __syncthreads();   // s_dk and J_k ready

        // ---- Inner L-BFGS solve ----
        int m_used = 0, newest = 0;

        // Initial cost + gradient
        float local_c = 0.0f;
        if (tid < T)
            sco_inner_costgrad_timestep(
                tid, T, n_act, s_traj, s_qk, s_dk, J_k,
                s_lower, s_upper,
                w_smooth, w_acc, w_jerk, w_limits, w_trust, w_coll,
                collision_margin, &local_c, grad);
        float best_cost = block_reduce_sum(local_c, smem_r, tid, bdim);

        // Save best_x, g_prev
        for (int i = tid; i < n; i += bdim) {
            best_x[i] = s_traj[i];
            g_prev[i] = grad[i];
        }

        for (int iter = 0; iter < n_inner_iters; iter++) {

            // ---- Cost + gradient at current s_traj ----
            local_c = 0.0f;
            if (tid < T)
                sco_inner_costgrad_timestep(
                    tid, T, n_act, s_traj, s_qk, s_dk, J_k,
                    s_lower, s_upper,
                    w_smooth, w_acc, w_jerk, w_limits, w_trust, w_coll,
                    collision_margin, &local_c, grad);
            float cost_val = block_reduce_sum(local_c, smem_r, tid, bdim);

            // ---- L-BFGS curvature update ----
            if (iter > 0) {
                // dir[] holds s_k from previous iteration
                float loc_sy = 0.0f, loc_yy = 0.0f;
                if (tid < T) {
                    for (int d = 0; d < n_act; d++) {
                        int i = tid * n_act + d;
                        float yi = grad[i] - g_prev[i];
                        loc_sy += dir[i] * yi;
                        loc_yy += yi * yi;
                    }
                }
                float sy = block_reduce_sum(loc_sy, smem_r, tid, bdim);
                float yy = block_reduce_sum(loc_yy, smem_r, tid, bdim);

                bool valid = (sy > 1e-10f * yy + 1e-30f);
                if (valid) {
                    int nn = (newest + 1) % m_lbfgs;
                    if (tid < T) {
                        for (int d = 0; d < n_act; d++) {
                            int i = tid * n_act + d;
                            s_lbfgs[nn*n + i] = dir[i];
                            y_lbfgs[nn*n + i] = grad[i] - g_prev[i];
                        }
                    }
                    if (tid == 0) rho_buf[nn] = 1.0f / (sy + 1e-30f);
                    newest = nn;
                    if (m_used < m_lbfgs) m_used++;
                    __syncthreads();
                }
            }

            // Save g_prev
            if (tid < T)
                for (int d = 0; d < n_act; d++)
                    g_prev[tid*n_act+d] = grad[tid*n_act+d];

            // ---- L-BFGS direction ----
            sco_lbfgs_two_loop_coop(
                grad, s_lbfgs, y_lbfgs, rho_buf, alpha_hist, dir,
                T, n_act, m_lbfgs, m_used, newest,
                tid, bdim, smem_r);

            // Pin endpoints in direction
            if (tid < n_act) {
                dir[tid]               = 0.0f;
                dir[(T-1)*n_act + tid] = 0.0f;
            }

            // ---- Parallel line search (5 alphas, parallel cost eval) ----
            const float LS_A[5] = {1.0f, 0.5f, 0.25f, 0.1f, 0.025f};
            float trial_c[5];

            // Compute all 5 trial costs with parallel per-timestep evaluation
            for (int ai = 0; ai < 5; ai++) {
                float lc = 0.0f;
                if (tid < T)
                    lc = sco_trial_cost_timestep(
                        tid, LS_A[ai], T, n_act,
                        s_traj, s_qk, s_dk, J_k, dir,
                        s_lower, s_upper,
                        w_smooth, w_acc, w_jerk, w_limits, w_trust, w_coll,
                        collision_margin);
                trial_c[ai] = block_reduce_sum(lc, smem_r, tid, bdim);
            }

            // Pick best alpha (sufficient decrease or lowest cost)
            float suff = cost_val * (1.0f - 1e-4f);
            int best_ai = 0;
            float best_ls = trial_c[0];
            for (int ai = 1; ai < 5; ai++) {
                if (trial_c[ai] < best_ls) { best_ls = trial_c[ai]; best_ai = ai; }
            }
            for (int ai = 0; ai < 5; ai++) {
                if (trial_c[ai] < suff) { best_ai = ai; best_ls = trial_c[ai]; break; }
            }

            // ---- Update traj + save s_k ----
            float alpha_star = LS_A[best_ai];
            if (tid < T) {
                for (int d = 0; d < n_act; d++) {
                    int i = tid * n_act + d;
                    float step = alpha_star * dir[i];
                    s_traj[i] += step;
                    dir[i] = step;   // s_k for next curvature update
                }
            }
            __syncthreads();

            // Track best iterate
            if (best_ls < best_cost) {
                for (int i = tid; i < n; i += bdim) best_x[i] = s_traj[i];
                best_cost = best_ls;
            }
        } // end inner loop

        // Restore best iterate
        for (int i = tid; i < n; i += bdim) s_traj[i] = best_x[i];
        __syncthreads();

        // Re-pin endpoints
        if (tid < n_act) {
            s_traj[tid]               = s_start[tid];
            s_traj[(T-1)*n_act + tid] = s_goal[tid];
        }
        __syncthreads();

        w_coll = fminf(w_coll * penalty_scale, w_collision_max);
    } // end outer loop

    // ==== Final nonlinear cost evaluation (cooperative) ====
    float nl_cost = 0.0f;
    if (tid < T) {
        float* my_Tw = T_world_pool + tid * n_joints * 7;
        const float* q_t = s_traj + tid * n_act;

        fk_single(q_t, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                  s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                  my_Tw, n_joints, n_act);

        // Self-collision pairs (actual distances, not smooth-min)
        for (int p = 0; p < P; p++) {
            int li = s_pair_i[p], lj = s_pair_j[p];
            float min_d = 1e10f;
            for (int si = 0; si < S; si++) {
                float ri = s_sphere_rad[li*S+si];
                if (ri < 0.0f) continue;
                float lci[3] = {s_sphere_off[(li*S+si)*3],
                                s_sphere_off[(li*S+si)*3+1],
                                s_sphere_off[(li*S+si)*3+2]};
                float ci[3]; sco_apply_se3(my_Tw + li*7, lci, ci);
                for (int sj = 0; sj < S; sj++) {
                    float rj = s_sphere_rad[lj*S+sj];
                    if (rj < 0.0f) continue;
                    float lcj[3] = {s_sphere_off[(lj*S+sj)*3],
                                    s_sphere_off[(lj*S+sj)*3+1],
                                    s_sphere_off[(lj*S+sj)*3+2]};
                    float cj[3]; sco_apply_se3(my_Tw + lj*7, lcj, cj);
                    float d = sco_sphere_sphere_dist(
                        ci[0],ci[1],ci[2],ri, cj[0],cj[1],cj[2],rj);
                    if (d < min_d) min_d = d;
                }
            }
            if (min_d < 1e9f)
                nl_cost -= fminf(sco_colldist_from_sdf(min_d, collision_margin), 0.0f)
                           * w_collision_max;
        }

        // World collision — hard min over S spheres per (link, obstacle) pair,
        // then apply colldist_from_sdf once (matches JAX's compute_world_collision_distance
        // which calls collide_link_vs_world → dist_spheres.min(axis=0)).
        // Compute sphere positions on the fly (no local cache array) to avoid
        // CUDA local memory addressing issues with 2D arrays in nested loops.
        for (int j = 0; j < N; j++) {
            for (int m = 0; m < Ms; m++) {
                float min_d = 1e10f;
                for (int s = 0; s < S; s++) {
                    float r = s_sphere_rad[j*S+s];
                    if (r < 0.0f) continue;
                    float lc[3] = {s_sphere_off[(j*S+s)*3],
                                   s_sphere_off[(j*S+s)*3+1],
                                   s_sphere_off[(j*S+s)*3+2]};
                    float wc[3]; sco_apply_se3(my_Tw + j*7, lc, wc);
                    float d = sco_sphere_sphere_dist(
                        wc[0],wc[1],wc[2],r,
                        world_spheres[m*4],world_spheres[m*4+1],
                        world_spheres[m*4+2],world_spheres[m*4+3]);
                    if (d < min_d) min_d = d;
                }
                if (min_d < 1e9f)
                    nl_cost -= fminf(sco_colldist_from_sdf(min_d, collision_margin), 0.0f)
                               * w_collision_max;
            }
            for (int m = 0; m < Mc; m++) {
                const float* cap = world_capsules + m*7;
                float min_d = 1e10f;
                for (int s = 0; s < S; s++) {
                    float r = s_sphere_rad[j*S+s];
                    if (r < 0.0f) continue;
                    float lc[3] = {s_sphere_off[(j*S+s)*3],
                                   s_sphere_off[(j*S+s)*3+1],
                                   s_sphere_off[(j*S+s)*3+2]};
                    float wc[3]; sco_apply_se3(my_Tw + j*7, lc, wc);
                    float d = sco_sphere_capsule_dist(
                        wc[0],wc[1],wc[2],r,
                        cap[0],cap[1],cap[2],cap[3],cap[4],cap[5],cap[6]);
                    if (d < min_d) min_d = d;
                }
                if (min_d < 1e9f)
                    nl_cost -= fminf(sco_colldist_from_sdf(min_d, collision_margin), 0.0f)
                               * w_collision_max;
            }
            for (int m = 0; m < Mb; m++) {
                const float* bx = world_boxes + m*15;
                float min_d = 1e10f;
                for (int s = 0; s < S; s++) {
                    float r = s_sphere_rad[j*S+s];
                    if (r < 0.0f) continue;
                    float lc[3] = {s_sphere_off[(j*S+s)*3],
                                   s_sphere_off[(j*S+s)*3+1],
                                   s_sphere_off[(j*S+s)*3+2]};
                    float wc[3]; sco_apply_se3(my_Tw + j*7, lc, wc);
                    float d = sco_sphere_box_dist(
                        wc[0],wc[1],wc[2],r,
                        bx[0],bx[1],bx[2], bx[3],bx[4],bx[5],
                        bx[6],bx[7],bx[8], bx[9],bx[10],bx[11],
                        bx[12],bx[13],bx[14]);
                    if (d < min_d) min_d = d;
                }
                if (min_d < 1e9f)
                    nl_cost -= fminf(sco_colldist_from_sdf(min_d, collision_margin), 0.0f)
                               * w_collision_max;
            }
            for (int m = 0; m < Mh; m++) {
                const float* hs = world_halfspaces + m*6;
                float min_d = 1e10f;
                for (int s = 0; s < S; s++) {
                    float r = s_sphere_rad[j*S+s];
                    if (r < 0.0f) continue;
                    float lc[3] = {s_sphere_off[(j*S+s)*3],
                                   s_sphere_off[(j*S+s)*3+1],
                                   s_sphere_off[(j*S+s)*3+2]};
                    float wc[3]; sco_apply_se3(my_Tw + j*7, lc, wc);
                    float d = sco_sphere_halfspace_dist(
                        wc[0],wc[1],wc[2],r,
                        hs[0],hs[1],hs[2],hs[3],hs[4],hs[5]);
                    if (d < min_d) min_d = d;
                }
                if (min_d < 1e9f)
                    nl_cost -= fminf(sco_colldist_from_sdf(min_d, collision_margin), 0.0f)
                               * w_collision_max;
            }
        }

        // Smoothness + limits (per-timestep contributions)
        {
            const float ST[5] = {-1.f/12.f, 16.f/12.f, -30.f/12.f, 16.f/12.f, -1.f/12.f};
            int n_acc = (T >= 5) ? T-4 : 0, n_jerk = (T >= 6) ? T-5 : 0;
            if (tid < n_acc) {
                for (int d = 0; d < n_act; d++) {
                    float a = 0.0f;
                    for (int k = 0; k < 5; k++) a += ST[k] * s_traj[(tid+k)*n_act+d];
                    nl_cost += w_smooth * w_acc * a * a;
                }
            }
            if (tid < n_jerk) {
                for (int d = 0; d < n_act; d++) {
                    float a0=0,a1=0;
                    for (int k=0;k<5;k++) {
                        a0 += ST[k]*s_traj[(tid+k)*n_act+d];
                        a1 += ST[k]*s_traj[(tid+1+k)*n_act+d];
                    }
                    float j=a1-a0; nl_cost += w_smooth * w_jerk * j * j;
                }
            }
            for (int d = 0; d < n_act; d++) {
                float qv = s_traj[tid*n_act+d];
                float viol = fmaxf(0.0f,qv-s_upper[d])+fmaxf(0.0f,s_lower[d]-qv);
                nl_cost += w_limits * viol * viol;
            }
        }
    }

    float final_cost = block_reduce_sum(nl_cost, smem_r, tid, bdim);

    // ==== Write outputs ====
    for (int i = tid; i < n; i += bdim)
        out_trajs[b * n + i] = s_traj[i];
    if (tid == 0)
        out_costs[b] = final_cost;
}

// ---------------------------------------------------------------------------
// XLA FFI handler — with CUDA graph caching
// ---------------------------------------------------------------------------

static ScoTrajoptGraphCache s_trajopt_cache;

static ffi::Error ScoTrajoptCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> init_trajs,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::F32> sphere_offsets,
    ffi::Buffer<ffi::DataType::F32> sphere_radii,
    ffi::Buffer<ffi::DataType::S32> pair_i_buf,
    ffi::Buffer<ffi::DataType::S32> pair_j_buf,
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::F32> start,
    ffi::Buffer<ffi::DataType::F32> goal,
    int64_t n_outer_iters,
    int64_t n_inner_iters,
    int64_t m_lbfgs,
    int64_t S,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_trust,
    float w_collision, float w_collision_max, float penalty_scale,
    float collision_margin, float smooth_min_temperature, float fd_eps,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_trajs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_costs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_workspace)
{
    const int B        = static_cast<int>(init_trajs.dimensions()[0]);
    const int T        = static_cast<int>(init_trajs.dimensions()[1]);
    const int n_act    = static_cast<int>(init_trajs.dimensions()[2]);
    const int n_joints = static_cast<int>(twists.dimensions()[0]);
    const int N        = static_cast<int>(sphere_offsets.dimensions()[0])
                         / (static_cast<int>(S) * 3);
    const int P  = (pair_i_buf.dimensions().size() > 0)
                   ? static_cast<int>(pair_i_buf.dimensions()[0]) : 0;
    const int Ms = (world_spheres.dimensions().size() > 0)
                   ? static_cast<int>(world_spheres.dimensions()[0]) : 0;
    const int Mc = (world_capsules.dimensions().size() > 0)
                   ? static_cast<int>(world_capsules.dimensions()[0]) : 0;
    const int Mb = (world_boxes.dimensions().size() > 0)
                   ? static_cast<int>(world_boxes.dimensions()[0]) : 0;
    const int Mh = (world_halfspaces.dimensions().size() > 0)
                   ? static_cast<int>(world_halfspaces.dimensions()[0]) : 0;

    // Block dimension: next power-of-2 >= T, at least 32, at most 256
    int block_dim = 32;
    while (block_dim < T) block_dim *= 2;
    if (block_dim > 256) block_dim = 256;

    const int n = T * n_act;
    const int m = static_cast<int>(m_lbfgs);

    // Per-block workspace layout (floats):
    //   grad[n], dir[n], best_x[n], g_prev[n],
    //   J_k[T*G*n_act], s_lbfgs[m*n], y_lbfgs[m*n],
    //   rho_buf[SCO_MAX_M], alpha_hist[SCO_MAX_M],
    //   T_world_pool[T * n_joints * 7]
    const int workspace_stride =
        4 * n
        + T * SCO_MAX_G * n_act
        + 2 * m * n
        + 2 * SCO_MAX_M
        + T * n_joints * 7;

    // Dynamic shared memory: s_traj[n] + s_qk[n] + s_dk[T*G] + smem_r[block_dim]
    const int dyn_smem_bytes =
        (2 * n + T * SCO_MAX_G + block_dim) * static_cast<int>(sizeof(float));

    // Build the kernel argument array (pointers to each argument).
    const float* p_init_trajs      = init_trajs.typed_data();
    const float* p_twists          = twists.typed_data();
    const float* p_parent_tf       = parent_tf.typed_data();
    const int*   p_parent_idx      = parent_idx.typed_data();
    const int*   p_act_idx         = act_idx.typed_data();
    const float* p_mimic_mul       = mimic_mul.typed_data();
    const float* p_mimic_off       = mimic_off.typed_data();
    const int*   p_mimic_act_idx   = mimic_act_idx.typed_data();
    const int*   p_topo_inv        = topo_inv.typed_data();
    const float* p_sphere_offsets  = sphere_offsets.typed_data();
    const float* p_sphere_radii    = sphere_radii.typed_data();
    const int*   p_pair_i          = pair_i_buf.typed_data();
    const int*   p_pair_j          = pair_j_buf.typed_data();
    const float* p_world_spheres   = world_spheres.typed_data();
    const float* p_world_capsules  = world_capsules.typed_data();
    const float* p_world_boxes     = world_boxes.typed_data();
    const float* p_world_halfspaces= world_halfspaces.typed_data();
    const float* p_lower           = lower.typed_data();
    const float* p_upper           = upper.typed_data();
    const float* p_start           = start.typed_data();
    const float* p_goal            = goal.typed_data();
    float*       p_out_trajs       = out_trajs->typed_data();
    float*       p_out_costs       = out_costs->typed_data();
    float*       p_out_workspace   = out_workspace->typed_data();
    int          k_workspace_stride= workspace_stride;
    int          k_B = B, k_T = T, k_n_joints = n_joints, k_n_act = n_act;
    int          k_N = N, k_S = static_cast<int>(S), k_P = P;
    int          k_Ms = Ms, k_Mc = Mc, k_Mb = Mb, k_Mh = Mh;
    int          k_n_outer = static_cast<int>(n_outer_iters);
    int          k_n_inner = static_cast<int>(n_inner_iters);
    int          k_m_lbfgs = static_cast<int>(m_lbfgs);

    void* kargs[] = {
        &p_init_trajs, &p_twists, &p_parent_tf, &p_parent_idx, &p_act_idx,
        &p_mimic_mul, &p_mimic_off, &p_mimic_act_idx, &p_topo_inv,
        &p_sphere_offsets, &p_sphere_radii,
        &p_pair_i, &p_pair_j,
        &p_world_spheres, &p_world_capsules, &p_world_boxes, &p_world_halfspaces,
        &p_lower, &p_upper, &p_start, &p_goal,
        &p_out_trajs, &p_out_costs, &p_out_workspace,
        &k_workspace_stride,
        &k_B, &k_T, &k_n_joints, &k_n_act,
        &k_N, &k_S, &k_P, &k_Ms, &k_Mc, &k_Mb, &k_Mh,
        &k_n_outer, &k_n_inner, &k_m_lbfgs,
        &w_smooth, &w_acc, &w_jerk,
        &w_limits, &w_trust,
        &w_collision, &w_collision_max, &penalty_scale,
        &collision_margin, &smooth_min_temperature, &fd_eps,
    };

    if (!s_trajopt_cache.shape_matches(B, T, n_act, n_joints, N, k_S, P,
                                        Ms, Mc, Mb, Mh)) {
        // Shape changed — recapture the graph.
        s_trajopt_cache.invalidate();
        cudaError_t e = s_trajopt_cache.ensure_capture_stream();
        if (e != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

        e = cudaStreamBeginCapture(s_trajopt_cache.capture_stream,
                                   cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

        sco_trajopt_kernel<<<B, block_dim, dyn_smem_bytes,
                             s_trajopt_cache.capture_stream>>>(
            p_init_trajs, p_twists, p_parent_tf, p_parent_idx, p_act_idx,
            p_mimic_mul, p_mimic_off, p_mimic_act_idx, p_topo_inv,
            p_sphere_offsets, p_sphere_radii,
            p_pair_i, p_pair_j,
            p_world_spheres, p_world_capsules, p_world_boxes, p_world_halfspaces,
            p_lower, p_upper, p_start, p_goal,
            p_out_trajs, p_out_costs, p_out_workspace,
            k_workspace_stride,
            k_B, k_T, k_n_joints, k_n_act,
            k_N, k_S, k_P, k_Ms, k_Mc, k_Mb, k_Mh,
            k_n_outer, k_n_inner, k_m_lbfgs,
            w_smooth, w_acc, w_jerk,
            w_limits, w_trust,
            w_collision, w_collision_max, penalty_scale,
            collision_margin, smooth_min_temperature, fd_eps);

        e = cudaStreamEndCapture(s_trajopt_cache.capture_stream,
                                 &s_trajopt_cache.graph);
        if (e != cudaSuccess) {
            s_trajopt_cache.invalidate();
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
        }

        e = s_trajopt_cache.finalize_capture(B, T, n_act, n_joints,
                                              N, k_S, P, Ms, Mc, Mb, Mh);
        if (e != cudaSuccess) {
            s_trajopt_cache.invalidate();
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
        }
    } else {
        // Shape unchanged — update kernel node params (new buffer pointers / scalars).
        cudaKernelNodeParams kp = {};
        kp.func           = s_trajopt_cache.func_ptr;
        kp.gridDim        = s_trajopt_cache.grid_dim;
        kp.blockDim       = s_trajopt_cache.block_dim;
        kp.sharedMemBytes = s_trajopt_cache.shared_mem;
        kp.kernelParams   = kargs;
        kp.extra          = nullptr;

        cudaError_t e = cudaGraphExecKernelNodeSetParams(
            s_trajopt_cache.exec, s_trajopt_cache.kernel_node, &kp);
        if (e != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    // Launch the cached graph on the XLA stream.
    cudaError_t e = cudaGraphLaunch(s_trajopt_cache.exec, stream);
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScoTrajoptCudaFfi, ScoTrajoptCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // init_trajs
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sphere_offsets
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sphere_radii
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // pair_j
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // start
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // goal
        .Attr<int64_t>("n_outer_iters")
        .Attr<int64_t>("n_inner_iters")
        .Attr<int64_t>("m_lbfgs")
        .Attr<int64_t>("S")
        .Attr<float>("w_smooth")
        .Attr<float>("w_acc")
        .Attr<float>("w_jerk")
        .Attr<float>("w_limits")
        .Attr<float>("w_trust")
        .Attr<float>("w_collision")
        .Attr<float>("w_collision_max")
        .Attr<float>("penalty_scale")
        .Attr<float>("collision_margin")
        .Attr<float>("smooth_min_temperature")
        .Attr<float>("fd_eps")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out_trajs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out_costs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out_workspace
);
