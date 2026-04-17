/**
 * CHOMP TrajOpt CUDA kernel.
 *
 * This backend implements CHOMP-style iterative trajectory updates on GPU with:
 *   - smoothness (acc + jerk) objective
 *   - soft joint limits
 *   - nonlinear collision hinge penalties
 *   - endpoint pinning
 *   - per-iteration line search
 *
 * The implementation intentionally prioritizes correctness and API parity over
 * maximal throughput. Each trajectory is assigned to one CUDA block and the
 * first thread in each block performs the optimization loop.
 */

#include "_ik_cuda_helpers.cuh"
#include "_collision_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

#ifndef CHOMP_MAX_T
#define CHOMP_MAX_T 128
#endif

#ifndef CHOMP_MAX_DOF
#define CHOMP_MAX_DOF MAX_ACT
#endif

#ifndef CHOMP_MAX_N
#define CHOMP_MAX_N MAX_JOINTS
#endif

#ifndef CHOMP_MAX_S
#define CHOMP_MAX_S 8
#endif

#ifndef CHOMP_MAX_PAIRS
#define CHOMP_MAX_PAIRS 256
#endif

// ---------------------------------------------------------------------------
// Geometry distance helpers (shared)
// ---------------------------------------------------------------------------

#define chomp_sphere_sphere_dist sphere_sphere_dist
#define chomp_sphere_capsule_dist sphere_capsule_dist
#define chomp_sphere_box_dist sphere_box_dist
#define chomp_sphere_halfspace_dist sphere_halfspace_dist
#define chomp_apply_se3 apply_se3_point

// ---------------------------------------------------------------------------
// Cost helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float chomp_xv(
    const float* __restrict__ traj,
    const float* __restrict__ dir,
    int use_trial,
    float alpha,
    int idx)
{
    return use_trial ? (traj[idx] + alpha * dir[idx]) : traj[idx];
}

static __device__ float chomp_collision_cost_cfg(
    const float* __restrict__ cfg,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int* __restrict__ parent_idx,
    const int* __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int* __restrict__ mimic_act_idx,
    const int* __restrict__ topo_inv,
    const float* __restrict__ sphere_offsets,
    const float* __restrict__ sphere_radii,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    int n_joints,
    int n_act,
    int N,
    int S,
    int P,
    int Ms,
    int Mc,
    int Mb,
    int Mh,
    float collision_margin,
    float* __restrict__ T_world)
{
    fk_single(cfg, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv,
              T_world, n_joints, n_act);

    float cost = 0.0f;

    // Self-collision: one distance per active link pair (minimum over spheres).
    for (int p = 0; p < P; p++) {
        int li = pair_i[p], lj = pair_j[p];
        float min_d = 1e10f;
        for (int si = 0; si < S; si++) {
            float ri = sphere_radii[li * S + si];
            if (ri < 0.0f) continue;
            float lci[3] = {
                sphere_offsets[(li * S + si) * 3 + 0],
                sphere_offsets[(li * S + si) * 3 + 1],
                sphere_offsets[(li * S + si) * 3 + 2],
            };
            float ci[3];
            chomp_apply_se3(T_world + li * 7, lci, ci);

            for (int sj = 0; sj < S; sj++) {
                float rj = sphere_radii[lj * S + sj];
                if (rj < 0.0f) continue;
                float lcj[3] = {
                    sphere_offsets[(lj * S + sj) * 3 + 0],
                    sphere_offsets[(lj * S + sj) * 3 + 1],
                    sphere_offsets[(lj * S + sj) * 3 + 2],
                };
                float cj[3];
                chomp_apply_se3(T_world + lj * 7, lcj, cj);

                float d = chomp_sphere_sphere_dist(
                    ci[0], ci[1], ci[2], ri,
                    cj[0], cj[1], cj[2], rj);
                if (d < min_d) min_d = d;
            }
        }
        if (min_d < 1e9f) {
            float v = fmaxf(0.0f, collision_margin - min_d);
            cost += v * v;
        }
    }

    // World collision: hard min over link spheres per (link, obstacle) pair.
    for (int j = 0; j < N; j++) {
        for (int m = 0; m < Ms; m++) {
            float min_d = 1e10f;
            for (int s = 0; s < S; s++) {
                float r = sphere_radii[j * S + s];
                if (r < 0.0f) continue;
                float lc[3] = {
                    sphere_offsets[(j * S + s) * 3 + 0],
                    sphere_offsets[(j * S + s) * 3 + 1],
                    sphere_offsets[(j * S + s) * 3 + 2],
                };
                float wc[3];
                chomp_apply_se3(T_world + j * 7, lc, wc);
                float d = chomp_sphere_sphere_dist(
                    wc[0], wc[1], wc[2], r,
                    world_spheres[m * 4 + 0], world_spheres[m * 4 + 1],
                    world_spheres[m * 4 + 2], world_spheres[m * 4 + 3]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) {
                float v = fmaxf(0.0f, collision_margin - min_d);
                cost += v * v;
            }
        }

        for (int m = 0; m < Mc; m++) {
            float min_d = 1e10f;
            const float* cap = world_capsules + m * 7;
            for (int s = 0; s < S; s++) {
                float r = sphere_radii[j * S + s];
                if (r < 0.0f) continue;
                float lc[3] = {
                    sphere_offsets[(j * S + s) * 3 + 0],
                    sphere_offsets[(j * S + s) * 3 + 1],
                    sphere_offsets[(j * S + s) * 3 + 2],
                };
                float wc[3];
                chomp_apply_se3(T_world + j * 7, lc, wc);
                float d = chomp_sphere_capsule_dist(
                    wc[0], wc[1], wc[2], r,
                    cap[0], cap[1], cap[2], cap[3], cap[4], cap[5], cap[6]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) {
                float v = fmaxf(0.0f, collision_margin - min_d);
                cost += v * v;
            }
        }

        for (int m = 0; m < Mb; m++) {
            float min_d = 1e10f;
            const float* bx = world_boxes + m * 15;
            for (int s = 0; s < S; s++) {
                float r = sphere_radii[j * S + s];
                if (r < 0.0f) continue;
                float lc[3] = {
                    sphere_offsets[(j * S + s) * 3 + 0],
                    sphere_offsets[(j * S + s) * 3 + 1],
                    sphere_offsets[(j * S + s) * 3 + 2],
                };
                float wc[3];
                chomp_apply_se3(T_world + j * 7, lc, wc);
                float d = chomp_sphere_box_dist(
                    wc[0], wc[1], wc[2], r,
                    bx[0], bx[1], bx[2],
                    bx[3], bx[4], bx[5],
                    bx[6], bx[7], bx[8],
                    bx[9], bx[10], bx[11],
                    bx[12], bx[13], bx[14]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) {
                float v = fmaxf(0.0f, collision_margin - min_d);
                cost += v * v;
            }
        }

        for (int m = 0; m < Mh; m++) {
            float min_d = 1e10f;
            const float* hs = world_halfspaces + m * 6;
            for (int s = 0; s < S; s++) {
                float r = sphere_radii[j * S + s];
                if (r < 0.0f) continue;
                float lc[3] = {
                    sphere_offsets[(j * S + s) * 3 + 0],
                    sphere_offsets[(j * S + s) * 3 + 1],
                    sphere_offsets[(j * S + s) * 3 + 2],
                };
                float wc[3];
                chomp_apply_se3(T_world + j * 7, lc, wc);
                float d = chomp_sphere_halfspace_dist(
                    wc[0], wc[1], wc[2], r,
                    hs[0], hs[1], hs[2], hs[3], hs[4], hs[5]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) {
                float v = fmaxf(0.0f, collision_margin - min_d);
                cost += v * v;
            }
        }
    }

    return cost;
}

static __device__ float chomp_total_cost(
    const float* __restrict__ traj,
    const float* __restrict__ dir,
    int use_trial,
    float alpha,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int* __restrict__ parent_idx,
    const int* __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int* __restrict__ mimic_act_idx,
    const int* __restrict__ topo_inv,
    const float* __restrict__ sphere_offsets,
    const float* __restrict__ sphere_radii,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    int T,
    int n_joints,
    int n_act,
    int N,
    int S,
    int P,
    int Ms,
    int Mc,
    int Mb,
    int Mh,
    float w_smooth,
    float w_acc,
    float w_jerk,
    float w_limits,
    float w_collision,
    float collision_margin,
    float* __restrict__ tw_buffer)
{
    const float ST[5] = {-1.f / 12.f, 16.f / 12.f, -30.f / 12.f, 16.f / 12.f, -1.f / 12.f};

    float cost = 0.0f;
    int n_acc = (T >= 5) ? (T - 4) : 0;
    int n_jerk = (T >= 6) ? (T - 5) : 0;

    // Smoothness.
    for (int t = 0; t < n_acc; t++) {
        for (int d = 0; d < n_act; d++) {
            float a = 0.0f;
            for (int k = 0; k < 5; k++) {
                int idx = (t + k) * n_act + d;
                a += ST[k] * chomp_xv(traj, dir, use_trial, alpha, idx);
            }
            cost += w_smooth * w_acc * a * a;
        }
    }
    for (int t = 0; t < n_jerk; t++) {
        for (int d = 0; d < n_act; d++) {
            float a0 = 0.0f, a1 = 0.0f;
            for (int k = 0; k < 5; k++) {
                int i0 = (t + k) * n_act + d;
                int i1 = (t + 1 + k) * n_act + d;
                a0 += ST[k] * chomp_xv(traj, dir, use_trial, alpha, i0);
                a1 += ST[k] * chomp_xv(traj, dir, use_trial, alpha, i1);
            }
            float j = a1 - a0;
            cost += w_smooth * w_jerk * j * j;
        }
    }

    // Limits.
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < n_act; d++) {
            int idx = t * n_act + d;
            float qv = chomp_xv(traj, dir, use_trial, alpha, idx);
            float viol = fmaxf(0.0f, qv - upper[d]) + fmaxf(0.0f, lower[d] - qv);
            cost += w_limits * viol * viol;
        }
    }

    // Collision (nonlinear).
    float cfg[CHOMP_MAX_DOF];
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < n_act; d++) {
            int idx = t * n_act + d;
            cfg[d] = chomp_xv(traj, dir, use_trial, alpha, idx);
        }
        float c_step = chomp_collision_cost_cfg(
            cfg,
            twists,
            parent_tf,
            parent_idx,
            act_idx,
            mimic_mul,
            mimic_off,
            mimic_act_idx,
            topo_inv,
            sphere_offsets,
            sphere_radii,
            pair_i,
            pair_j,
            world_spheres,
            world_capsules,
            world_boxes,
            world_halfspaces,
            n_joints,
            n_act,
            N,
            S,
            P,
            Ms,
            Mc,
            Mb,
            Mh,
            collision_margin,
            tw_buffer);
        cost += w_collision * c_step;
    }

    return cost;
}

static __device__ float chomp_smooth_limits_cost_and_grad(
    const float* __restrict__ traj,
    float* __restrict__ grad,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    int T,
    int n_act,
    float w_smooth,
    float w_acc,
    float w_jerk,
    float w_limits)
{
    const float ST[5] = {-1.f / 12.f, 16.f / 12.f, -30.f / 12.f, 16.f / 12.f, -1.f / 12.f};
    const int n = T * n_act;
    for (int i = 0; i < n; i++) grad[i] = 0.0f;

    float cost = 0.0f;
    int n_acc = (T >= 5) ? (T - 4) : 0;
    int n_jerk = (T >= 6) ? (T - 5) : 0;

    // Smoothness cost and gradient.
    for (int t = 0; t < n_acc; t++) {
        for (int d = 0; d < n_act; d++) {
            float a = 0.0f;
            for (int k = 0; k < 5; k++) a += ST[k] * traj[(t + k) * n_act + d];
            cost += w_smooth * w_acc * a * a;
        }
    }
    for (int t = 0; t < n_jerk; t++) {
        for (int d = 0; d < n_act; d++) {
            float a0 = 0.0f, a1 = 0.0f;
            for (int k = 0; k < 5; k++) {
                a0 += ST[k] * traj[(t + k) * n_act + d];
                a1 += ST[k] * traj[(t + 1 + k) * n_act + d];
            }
            float j = a1 - a0;
            cost += w_smooth * w_jerk * j * j;
        }
    }

    for (int sidx = 0; sidx < T; sidx++) {
        for (int d = 0; d < n_act; d++) {
            float g = 0.0f;

            int tlo = (sidx >= 4) ? (sidx - 4) : 0;
            int thi = (sidx < n_acc) ? sidx : (n_acc - 1);
            for (int tt = tlo; tt <= thi; tt++) {
                float a = 0.0f;
                for (int k = 0; k < 5; k++) a += ST[k] * traj[(tt + k) * n_act + d];
                g += 2.0f * w_acc * a * ST[sidx - tt];
            }

            int jlo = (sidx >= 5) ? (sidx - 5) : 0;
            int jhi = (sidx < n_jerk) ? sidx : (n_jerk - 1);
            for (int tt = jlo; tt <= jhi; tt++) {
                float a0 = 0.0f, a1 = 0.0f;
                for (int k = 0; k < 5; k++) {
                    a0 += ST[k] * traj[(tt + k) * n_act + d];
                    a1 += ST[k] * traj[(tt + 1 + k) * n_act + d];
                }
                float jk = a1 - a0;
                int k1 = sidx - tt - 1;
                int k2 = sidx - tt;
                float c1 = (k1 >= 0 && k1 <= 4) ? ST[k1] : 0.0f;
                float c2 = (k2 >= 0 && k2 <= 4) ? ST[k2] : 0.0f;
                g += 2.0f * w_jerk * jk * (c1 - c2);
            }

            grad[sidx * n_act + d] += w_smooth * g;
        }
    }

    // Limits cost and gradient.
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < n_act; d++) {
            int idx = t * n_act + d;
            float qv = traj[idx];
            float viol = fmaxf(0.0f, qv - upper[d]) + fmaxf(0.0f, lower[d] - qv);
            cost += w_limits * viol * viol;
            float sign = (qv > upper[d]) ? 1.0f : (qv < lower[d]) ? -1.0f : 0.0f;
            grad[idx] += 2.0f * w_limits * viol * sign;
        }
    }

    return cost;
}

__device__ __forceinline__ float chomp_metric_apply_at(
    const float* __restrict__ x,
    int i,
    int Tin,
    float reg)
{
    float y = 6.0f * x[i];
    if (i - 1 >= 0) y += -4.0f * x[i - 1];
    if (i + 1 < Tin) y += -4.0f * x[i + 1];
    if (i - 2 >= 0) y += x[i - 2];
    if (i + 2 < Tin) y += x[i + 2];
    return y + reg * x[i];
}

static __device__ void chomp_metric_cg_solve(
    const float* __restrict__ b,
    float* __restrict__ x,
    int Tin,
    float reg)
{
    if (Tin <= 0) return;

    float r[CHOMP_MAX_T];
    float p[CHOMP_MAX_T];
    float Ap[CHOMP_MAX_T];

    float rs_old = 0.0f;
    for (int i = 0; i < Tin; i++) {
        x[i] = 0.0f;
        r[i] = b[i];
        p[i] = r[i];
        rs_old += r[i] * r[i];
    }
    if (rs_old < 1e-20f) return;

    const int max_iters = (Tin < 24) ? Tin : 24;
    for (int it = 0; it < max_iters; it++) {
        float pAp = 0.0f;
        for (int i = 0; i < Tin; i++) {
            Ap[i] = chomp_metric_apply_at(p, i, Tin, reg);
            pAp += p[i] * Ap[i];
        }

        if (fabsf(pAp) < 1e-20f) break;

        const float alpha = rs_old / pAp;
        float rs_new = 0.0f;
        for (int i = 0; i < Tin; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rs_new += r[i] * r[i];
        }

        if (rs_new < 1e-12f) break;

        const float beta = rs_new / (rs_old + 1e-20f);
        for (int i = 0; i < Tin; i++) {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

__global__ void chomp_trajopt_kernel(
    const float* __restrict__ init_trajs,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int* __restrict__ parent_idx,
    const int* __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int* __restrict__ mimic_act_idx,
    const int* __restrict__ topo_inv,
    const float* __restrict__ sphere_offsets,
    const float* __restrict__ sphere_radii,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const float* __restrict__ start,
    const float* __restrict__ goal,
    float* __restrict__ out_trajs,
    float* __restrict__ out_costs,
    float* __restrict__ workspace,
    int workspace_stride,
    int B,
    int T,
    int n_joints,
    int n_act,
    int N,
    int S,
    int P,
    int Ms,
    int Mc,
    int Mb,
    int Mh,
    int n_iters,
    float step_size,
    float w_smooth,
    float w_acc,
    float w_jerk,
    float w_limits,
    float w_collision,
    float w_collision_max,
    float collision_penalty_scale,
    float collision_margin,
    int use_covariant_update,
    float smoothness_reg,
    float grad_clip_norm,
    float max_delta_per_step,
    int early_stop_patience,
    float min_cost_improve,
    float fd_eps)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    // Cache robot FK arrays in shared memory once per trajectory block.
    __shared__ float s_twists[CHOMP_MAX_N * 6];
    __shared__ float s_parent_tf[CHOMP_MAX_N * 7];
    __shared__ int s_parent_idx[CHOMP_MAX_N];
    __shared__ int s_act_idx[CHOMP_MAX_N];
    __shared__ float s_mimic_mul[CHOMP_MAX_N];
    __shared__ float s_mimic_off[CHOMP_MAX_N];
    __shared__ int s_mimic_act_idx[CHOMP_MAX_N];
    __shared__ int s_topo_inv[CHOMP_MAX_N];

    for (int i = threadIdx.x; i < n_joints * 6; i += blockDim.x) s_twists[i] = twists[i];
    for (int i = threadIdx.x; i < n_joints * 7; i += blockDim.x) s_parent_tf[i] = parent_tf[i];
    for (int i = threadIdx.x; i < n_joints; i += blockDim.x) {
        s_parent_idx[i] = parent_idx[i];
        s_act_idx[i] = act_idx[i];
        s_mimic_mul[i] = mimic_mul[i];
        s_mimic_off[i] = mimic_off[i];
        s_mimic_act_idx[i] = mimic_act_idx[i];
        s_topo_inv[i] = topo_inv[i];
    }
    __syncthreads();

    const int n = T * n_act;
    float* traj = out_trajs + b * n;

    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) traj[i] = init_trajs[b * n + i];
        for (int d = 0; d < n_act; d++) {
            traj[d] = start[d];
            traj[(T - 1) * n_act + d] = goal[d];
        }
    }
    __syncthreads();

    float* base = workspace + (size_t)b * workspace_stride;
    float* grad = base;
    float* dir = base + n;

    constexpr int LS_N = 4;
    const float alphas[LS_N] = {1.0f, 0.5f, 0.25f, 0.0f};
    __shared__ float s_w_coll;
    __shared__ float s_curr_cost;
    __shared__ float s_best_cost;
    __shared__ float s_best_alpha;
    __shared__ float s_trial_costs[LS_N];
    __shared__ int s_stall_steps;
    __shared__ float s_best_seen_cost;
    __shared__ int s_stop;
    __shared__ float s_cfg[CHOMP_MAX_DOF];
    __shared__ float s_collision_base;

    // Dynamic shared workspace for FK world-transform scratch buffers:
    // - FD gradients: n_act slots (one per perturbed joint)
    // - line search: LS_N slots (one per alpha candidate thread)
    extern __shared__ float s_dyn[];
    const int tw_stride = n_joints * 7;
    float* s_tw_fd = s_dyn;
    float* s_tw_ls = s_dyn + n_act * tw_stride;

    if (threadIdx.x == 0) {
        s_w_coll = w_collision;
        s_best_seen_cost = 1e30f;
        s_stall_steps = 0;
        s_stop = 0;
    }
    __syncthreads();

    for (int it = 0; it < n_iters; it++) {
        if (threadIdx.x == 0) {
            s_curr_cost = chomp_smooth_limits_cost_and_grad(
                traj,
                grad,
                lower,
                upper,
                T,
                n_act,
                w_smooth,
                w_acc,
                w_jerk,
                w_limits);
        }
        __syncthreads();

        // Collision cost + FD gradient. For each timestep, threads split DOFs.
        for (int t = 0; t < T; t++) {
            if (threadIdx.x == 0) {
                for (int d = 0; d < n_act; d++) {
                    s_cfg[d] = traj[t * n_act + d];
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                float tw_base[MAX_JOINTS * 7];
                s_collision_base = chomp_collision_cost_cfg(
                    s_cfg,
                    s_twists,
                    s_parent_tf,
                    s_parent_idx,
                    s_act_idx,
                    s_mimic_mul,
                    s_mimic_off,
                    s_mimic_act_idx,
                    s_topo_inv,
                    sphere_offsets,
                    sphere_radii,
                    pair_i,
                    pair_j,
                    world_spheres,
                    world_capsules,
                    world_boxes,
                    world_halfspaces,
                    n_joints,
                    n_act,
                    N,
                    S,
                    P,
                    Ms,
                    Mc,
                    Mb,
                    Mh,
                    collision_margin,
                    tw_base);
                s_curr_cost += s_w_coll * s_collision_base;
            }
            __syncthreads();

            for (int d = threadIdx.x; d < n_act; d += blockDim.x) {
                float q_pert[CHOMP_MAX_DOF];
                for (int j = 0; j < n_act; j++) q_pert[j] = s_cfg[j];

                float orig = q_pert[d];
                float* tw_fd = s_tw_fd + d * tw_stride;

                q_pert[d] = orig + fd_eps;
                float cp = chomp_collision_cost_cfg(
                    q_pert,
                    s_twists,
                    s_parent_tf,
                    s_parent_idx,
                    s_act_idx,
                    s_mimic_mul,
                    s_mimic_off,
                    s_mimic_act_idx,
                    s_topo_inv,
                    sphere_offsets,
                    sphere_radii,
                    pair_i,
                    pair_j,
                    world_spheres,
                    world_capsules,
                    world_boxes,
                    world_halfspaces,
                    n_joints,
                    n_act,
                    N,
                    S,
                    P,
                    Ms,
                    Mc,
                    Mb,
                    Mh,
                    collision_margin,
                    tw_fd);

                q_pert[d] = orig - fd_eps;
                float cm = chomp_collision_cost_cfg(
                    q_pert,
                    s_twists,
                    s_parent_tf,
                    s_parent_idx,
                    s_act_idx,
                    s_mimic_mul,
                    s_mimic_off,
                    s_mimic_act_idx,
                    s_topo_inv,
                    sphere_offsets,
                    sphere_radii,
                    pair_i,
                    pair_j,
                    world_spheres,
                    world_capsules,
                    world_boxes,
                    world_halfspaces,
                    n_joints,
                    n_act,
                    N,
                    S,
                    P,
                    Ms,
                    Mc,
                    Mb,
                    Mh,
                    collision_margin,
                    tw_fd);

                q_pert[d] = orig;

                const float inv2e = 1.0f / (2.0f * fd_eps);
                grad[t * n_act + d] += s_w_coll * (cp - cm) * inv2e;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            // Pin endpoints after all gradient terms are accumulated.
            for (int d = 0; d < n_act; d++) {
                grad[d] = 0.0f;
                grad[(T - 1) * n_act + d] = 0.0f;
            }

            if (s_curr_cost < s_best_seen_cost) {
                s_best_seen_cost = s_curr_cost;
            }

            if (grad_clip_norm > 0.0f) {
                float gg = 0.0f;
                for (int t = 1; t < T - 1; t++) {
                    for (int d = 0; d < n_act; d++) {
                        float g = grad[t * n_act + d];
                        gg += g * g;
                    }
                }

                float gnorm = sqrtf(gg);
                if (gnorm > grad_clip_norm) {
                    float s = grad_clip_norm / (gnorm + 1e-12f);
                    for (int t = 1; t < T - 1; t++) {
                        for (int d = 0; d < n_act; d++) {
                            grad[t * n_act + d] *= s;
                        }
                    }
                }
            }

            // Direction.
            const int Tin = (T > 2) ? (T - 2) : 0;
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < n_act; d++) {
                    int idx = t * n_act + d;
                    if (t == 0 || t == T - 1) {
                        dir[idx] = 0.0f;
                        continue;
                    }
                }
            }

            if (use_covariant_update && Tin > 0) {
                float g_int[CHOMP_MAX_T];
                float z_int[CHOMP_MAX_T];
                for (int d = 0; d < n_act; d++) {
                    for (int i = 0; i < Tin; i++) {
                        g_int[i] = grad[(i + 1) * n_act + d];
                    }
                    chomp_metric_cg_solve(g_int, z_int, Tin, smoothness_reg);
                    for (int i = 0; i < Tin; i++) {
                        dir[(i + 1) * n_act + d] = -z_int[i];
                    }
                }
            } else {
                for (int t = 1; t < T - 1; t++) {
                    for (int d = 0; d < n_act; d++) {
                        int idx = t * n_act + d;
                        dir[idx] = -grad[idx];
                    }
                }
            }

            s_best_cost = s_curr_cost;
            s_best_alpha = 0.0f;
        }
        __syncthreads();

        // Parallel line search: one thread per alpha candidate.
        if (threadIdx.x < LS_N) {
            float* tw_ls = s_tw_ls + threadIdx.x * tw_stride;
            float a = step_size * alphas[threadIdx.x];
            s_trial_costs[threadIdx.x] = chomp_total_cost(
                traj,
                dir,
                1,
                a,
                lower,
                upper,
                s_twists,
                s_parent_tf,
                s_parent_idx,
                s_act_idx,
                s_mimic_mul,
                s_mimic_off,
                s_mimic_act_idx,
                s_topo_inv,
                sphere_offsets,
                sphere_radii,
                pair_i,
                pair_j,
                world_spheres,
                world_capsules,
                world_boxes,
                world_halfspaces,
                T,
                n_joints,
                n_act,
                N,
                S,
                P,
                Ms,
                Mc,
                Mb,
                Mh,
                w_smooth,
                w_acc,
                w_jerk,
                w_limits,
                s_w_coll,
                collision_margin,
                tw_ls);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int ai = 0; ai < LS_N; ai++) {
                float c = s_trial_costs[ai];
                float a = step_size * alphas[ai];
                if (c < s_best_cost) {
                    s_best_cost = c;
                    s_best_alpha = a;
                }
            }

            if (s_best_alpha > 0.0f) {
                for (int t = 1; t < T - 1; t++) {
                    for (int d = 0; d < n_act; d++) {
                        int idx = t * n_act + d;
                        float step = s_best_alpha * dir[idx];
                        step = fmaxf(-max_delta_per_step, fminf(max_delta_per_step, step));
                        traj[idx] += step;
                    }
                }
            }

            bool improved = s_best_cost < (s_curr_cost - min_cost_improve);
            if (improved) {
                s_stall_steps = 0;
            } else {
                s_stall_steps += 1;
            }

            if (s_best_cost < s_best_seen_cost) {
                s_best_seen_cost = s_best_cost;
            }

            s_w_coll = fminf(s_w_coll * collision_penalty_scale, w_collision_max);

            for (int d = 0; d < n_act; d++) {
                traj[d] = start[d];
                traj[(T - 1) * n_act + d] = goal[d];
            }

            s_stop = (s_stall_steps >= early_stop_patience) ? 1 : 0;
        }
        __syncthreads();
        if (s_stop) break;
    }

    if (threadIdx.x == 0) {
        float tw_buffer[MAX_JOINTS * 7];
        float final_cost = chomp_total_cost(
            traj,
            nullptr,
            0,
            0.0f,
            lower,
            upper,
            s_twists,
            s_parent_tf,
            s_parent_idx,
            s_act_idx,
            s_mimic_mul,
            s_mimic_off,
            s_mimic_act_idx,
            s_topo_inv,
            sphere_offsets,
            sphere_radii,
            pair_i,
            pair_j,
            world_spheres,
            world_capsules,
            world_boxes,
            world_halfspaces,
            T,
            n_joints,
            n_act,
            N,
            S,
            P,
            Ms,
            Mc,
            Mb,
            Mh,
            w_smooth,
            w_acc,
            w_jerk,
            w_limits,
            w_collision_max,
            collision_margin,
            tw_buffer);

        out_costs[b] = final_cost;
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error ChompTrajoptCudaImpl(
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
    int64_t n_iters,
    int64_t S,
    float step_size,
    float w_smooth,
    float w_acc,
    float w_jerk,
    float w_limits,
    float w_collision,
    float w_collision_max,
    float collision_penalty_scale,
    float collision_margin,
    int64_t use_covariant_update,
    float smoothness_reg,
    float grad_clip_norm,
    float max_delta_per_step,
    int64_t early_stop_patience,
    float min_cost_improve,
    float fd_eps,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_trajs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_costs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_workspace)
{
    const int B = static_cast<int>(init_trajs.dimensions()[0]);
    const int T = static_cast<int>(init_trajs.dimensions()[1]);
    const int n_act = static_cast<int>(init_trajs.dimensions()[2]);
    const int n_joints = static_cast<int>(twists.dimensions()[0]);

    if (T > CHOMP_MAX_T || n_act > CHOMP_MAX_DOF || n_joints > CHOMP_MAX_N) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "CHOMP CUDA dimensions exceed compile-time limits."
        );
    }

    const int N = static_cast<int>(sphere_offsets.dimensions()[0])
                  / (static_cast<int>(S) * 3);
    const int P = (pair_i_buf.dimensions().size() > 0)
                  ? static_cast<int>(pair_i_buf.dimensions()[0]) : 0;
    const int Ms = (world_spheres.dimensions().size() > 0)
                   ? static_cast<int>(world_spheres.dimensions()[0]) : 0;
    const int Mc = (world_capsules.dimensions().size() > 0)
                   ? static_cast<int>(world_capsules.dimensions()[0]) : 0;
    const int Mb = (world_boxes.dimensions().size() > 0)
                   ? static_cast<int>(world_boxes.dimensions()[0]) : 0;
    const int Mh = (world_halfspaces.dimensions().size() > 0)
                   ? static_cast<int>(world_halfspaces.dimensions()[0]) : 0;

    const int n = T * n_act;
    const int workspace_stride = 3 * n;
    constexpr int LS_N = 4;
    const int dyn_shared_bytes = (n_act + LS_N) * n_joints * 7 * sizeof(float);

    // One warp is sufficient: n_act is small, and CHOMP work is dominated by
    // single-trajectory collision evaluations and synchronization.
    constexpr int THREADS = 32;

    chomp_trajopt_kernel<<<B, THREADS, dyn_shared_bytes, stream>>>(
        init_trajs.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        sphere_offsets.typed_data(),
        sphere_radii.typed_data(),
        pair_i_buf.typed_data(),
        pair_j_buf.typed_data(),
        world_spheres.typed_data(),
        world_capsules.typed_data(),
        world_boxes.typed_data(),
        world_halfspaces.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        start.typed_data(),
        goal.typed_data(),
        out_trajs->typed_data(),
        out_costs->typed_data(),
        out_workspace->typed_data(),
        workspace_stride,
        B,
        T,
        n_joints,
        n_act,
        N,
        static_cast<int>(S),
        P,
        Ms,
        Mc,
        Mb,
        Mh,
        static_cast<int>(n_iters),
        step_size,
        w_smooth,
        w_acc,
        w_jerk,
        w_limits,
        w_collision,
        w_collision_max,
        collision_penalty_scale,
        collision_margin,
        static_cast<int>(use_covariant_update),
        smoothness_reg,
        grad_clip_norm,
        max_delta_per_step,
        static_cast<int>(early_stop_patience),
        min_cost_improve,
        fd_eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ChompTrajoptCudaFfi,
    ChompTrajoptCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Attr<int64_t>("n_iters")
        .Attr<int64_t>("S")
        .Attr<float>("step_size")
        .Attr<float>("w_smooth")
        .Attr<float>("w_acc")
        .Attr<float>("w_jerk")
        .Attr<float>("w_limits")
        .Attr<float>("w_collision")
        .Attr<float>("w_collision_max")
        .Attr<float>("collision_penalty_scale")
        .Attr<float>("collision_margin")
        .Attr<int64_t>("use_covariant_update")
        .Attr<float>("smoothness_reg")
        .Attr<float>("grad_clip_norm")
        .Attr<float>("max_delta_per_step")
        .Attr<int64_t>("early_stop_patience")
        .Attr<float>("min_cost_improve")
        .Attr<float>("fd_eps")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());
