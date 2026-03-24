/**
 * Least-Squares TrajOpt CUDA kernel (single-thread per trajectory block).
 *
 * Formulates each linearized outer step as a large least-squares problem and
 * runs a diagonal-Gauss-Newton / LM update inside the kernel.
 *
 * Build with:
 *   bash src/pyronot/cuda_kernels/build_ls_trajopt_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

#ifndef LST_MAX_N
#define LST_MAX_N MAX_JOINTS
#endif

#ifndef LST_MAX_S
#define LST_MAX_S 32
#endif

#ifndef LST_MAX_PAIRS
#define LST_MAX_PAIRS 256
#endif

#ifndef LST_MAX_DOF
#define LST_MAX_DOF MAX_ACT
#endif

#define LST_G 5

__device__ __forceinline__ float lst_sphere_sphere_dist(
    float ax, float ay, float az, float ar,
    float bx, float by, float bz, float br)
{
    float dx = ax - bx;
    float dy = ay - by;
    float dz = az - bz;
    return sqrtf(dx * dx + dy * dy + dz * dz) - (ar + br);
}

__device__ __forceinline__ float lst_sphere_capsule_dist(
    float sx, float sy, float sz, float sr,
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr)
{
    float vx = x2 - x1;
    float vy = y2 - y1;
    float vz = z2 - z1;
    float len2 = vx * vx + vy * vy + vz * vz;
    float t = 0.0f;
    if (len2 > 1e-12f) {
        t = ((sx - x1) * vx + (sy - y1) * vy + (sz - z1) * vz) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    float cx = x1 + t * vx;
    float cy = y1 + t * vy;
    float cz = z1 + t * vz;
    float dx = sx - cx;
    float dy = sy - cy;
    float dz = sz - cz;
    return sqrtf(dx * dx + dy * dy + dz * dz) - (sr + cr);
}

__device__ __forceinline__ float lst_box_sdf_local(
    float p1, float p2, float p3,
    float hl1, float hl2, float hl3)
{
    float q1 = fabsf(p1) - hl1;
    float q2 = fabsf(p2) - hl2;
    float q3 = fabsf(p3) - hl3;
    float mq1 = fmaxf(q1, 0.0f);
    float mq2 = fmaxf(q2, 0.0f);
    float mq3 = fmaxf(q3, 0.0f);
    return sqrtf(mq1 * mq1 + mq2 * mq2 + mq3 * mq3)
           + fminf(fmaxf(fmaxf(q1, q2), q3), 0.0f);
}

__device__ __forceinline__ float lst_sphere_box_dist(
    float sx, float sy, float sz, float sr,
    float bcx, float bcy, float bcz,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float a3x, float a3y, float a3z,
    float hl1, float hl2, float hl3)
{
    float dx = sx - bcx;
    float dy = sy - bcy;
    float dz = sz - bcz;
    float p1 = dx * a1x + dy * a1y + dz * a1z;
    float p2 = dx * a2x + dy * a2y + dz * a2z;
    float p3 = dx * a3x + dy * a3y + dz * a3z;
    return lst_box_sdf_local(p1, p2, p3, hl1, hl2, hl3) - sr;
}

__device__ __forceinline__ float lst_sphere_halfspace_dist(
    float sx, float sy, float sz, float sr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    return (sx - px) * nx + (sy - py) * ny + (sz - pz) * nz - sr;
}

__device__ __forceinline__ void lst_apply_se3(
    const float* __restrict__ T,
    const float* __restrict__ p,
    float* __restrict__ out)
{
    quat_rotate(T, p, out);
    out[0] += T[4];
    out[1] += T[5];
    out[2] += T[6];
}

struct LstSmoothMinAcc {
    float max_val;
    float sum_exp;

    __device__ void init() {
        max_val = -1e30f;
        sum_exp = 0.0f;
    }

    __device__ void update(float d, float tau) {
        float v = -d / tau;
        if (v > max_val) {
            sum_exp *= expf(max_val - v);
            max_val = v;
        }
        sum_exp += expf(v - max_val);
    }

    __device__ float finalize(float tau) const {
        if (sum_exp <= 0.0f) {
            return 1e10f;
        }
        return -tau * (logf(sum_exp) + max_val);
    }
};

__device__ void lst_compute_coll_groups(
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
    float temperature,
    float* __restrict__ out_groups,
    float* __restrict__ T_world)
{
    fk_single(
        cfg,
        twists,
        parent_tf,
        parent_idx,
        act_idx,
        mimic_mul,
        mimic_off,
        mimic_act_idx,
        topo_inv,
        T_world,
        n_joints,
        n_act);

    LstSmoothMinAcc acc[LST_G];
    for (int g = 0; g < LST_G; g++) {
        acc[g].init();
    }

    for (int p = 0; p < P; p++) {
        int li = pair_i[p];
        int lj = pair_j[p];
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
            lst_apply_se3(T_world + li * 7, lci, ci);

            for (int sj = 0; sj < S; sj++) {
                float rj = sphere_radii[lj * S + sj];
                if (rj < 0.0f) continue;

                float lcj[3] = {
                    sphere_offsets[(lj * S + sj) * 3 + 0],
                    sphere_offsets[(lj * S + sj) * 3 + 1],
                    sphere_offsets[(lj * S + sj) * 3 + 2],
                };
                float cj[3];
                lst_apply_se3(T_world + lj * 7, lcj, cj);

                float d = lst_sphere_sphere_dist(
                    ci[0], ci[1], ci[2], ri,
                    cj[0], cj[1], cj[2], rj);
                if (d < min_d) min_d = d;
            }
        }

        if (min_d < 1e9f) {
            acc[0].update(min_d, temperature);
        }
    }

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
                lst_apply_se3(T_world + j * 7, lc, wc);

                float d = lst_sphere_sphere_dist(
                    wc[0], wc[1], wc[2], r,
                    world_spheres[m * 4 + 0],
                    world_spheres[m * 4 + 1],
                    world_spheres[m * 4 + 2],
                    world_spheres[m * 4 + 3]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) acc[1].update(min_d, temperature);
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
                lst_apply_se3(T_world + j * 7, lc, wc);

                float d = lst_sphere_capsule_dist(
                    wc[0], wc[1], wc[2], r,
                    cap[0], cap[1], cap[2],
                    cap[3], cap[4], cap[5], cap[6]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) acc[2].update(min_d, temperature);
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
                lst_apply_se3(T_world + j * 7, lc, wc);

                float d = lst_sphere_box_dist(
                    wc[0], wc[1], wc[2], r,
                    bx[0], bx[1], bx[2],
                    bx[3], bx[4], bx[5],
                    bx[6], bx[7], bx[8],
                    bx[9], bx[10], bx[11],
                    bx[12], bx[13], bx[14]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) acc[3].update(min_d, temperature);
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
                lst_apply_se3(T_world + j * 7, lc, wc);

                float d = lst_sphere_halfspace_dist(
                    wc[0], wc[1], wc[2], r,
                    hs[0], hs[1], hs[2],
                    hs[3], hs[4], hs[5]);
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9f) acc[4].update(min_d, temperature);
        }
    }

    for (int g = 0; g < LST_G; g++) {
        out_groups[g] = acc[g].finalize(temperature);
    }
}

__device__ int lst_build_residual(
    const float* __restrict__ traj,
    const float* __restrict__ qk,
    const float* __restrict__ dk,
    const float* __restrict__ Jk,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const float* __restrict__ start,
    const float* __restrict__ goal,
    int T,
    int n_act,
    float sqrt_w_acc,
    float sqrt_w_jerk,
    float sqrt_w_trust,
    float sqrt_w_limits,
    float sqrt_w_coll,
    float sqrt_w_endpoint,
    float collision_margin,
    float* __restrict__ out_r)
{
    int k = 0;

    for (int t = 0; t < T - 2; t++) {
        for (int d = 0; d < n_act; d++) {
            float acc = traj[(t + 2) * n_act + d]
                      - 2.0f * traj[(t + 1) * n_act + d]
                      + traj[t * n_act + d];
            out_r[k++] = sqrt_w_acc * acc;
        }
    }

    for (int t = 0; t < T - 3; t++) {
        for (int d = 0; d < n_act; d++) {
            float acc0 = traj[(t + 2) * n_act + d]
                       - 2.0f * traj[(t + 1) * n_act + d]
                       + traj[t * n_act + d];
            float acc1 = traj[(t + 3) * n_act + d]
                       - 2.0f * traj[(t + 2) * n_act + d]
                       + traj[(t + 1) * n_act + d];
            out_r[k++] = sqrt_w_jerk * (acc1 - acc0);
        }
    }

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < n_act; d++) {
            out_r[k++] = sqrt_w_trust * (traj[t * n_act + d] - qk[t * n_act + d]);
        }
    }

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < n_act; d++) {
            float v = fmaxf(0.0f, traj[t * n_act + d] - upper[d]);
            out_r[k++] = sqrt_w_limits * v;
        }
    }

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < n_act; d++) {
            float v = fmaxf(0.0f, lower[d] - traj[t * n_act + d]);
            out_r[k++] = sqrt_w_limits * v;
        }
    }

    for (int t = 0; t < T; t++) {
        for (int g = 0; g < LST_G; g++) {
            float d_lin = dk[t * LST_G + g];
            for (int d = 0; d < n_act; d++) {
                d_lin += Jk[(t * LST_G + g) * n_act + d]
                      * (traj[t * n_act + d] - qk[t * n_act + d]);
            }
            float viol = fmaxf(0.0f, collision_margin - d_lin);
            out_r[k++] = sqrt_w_coll * viol;
        }
    }

    for (int d = 0; d < n_act; d++) {
        out_r[k++] = sqrt_w_endpoint * (traj[d] - start[d]);
    }
    for (int d = 0; d < n_act; d++) {
        out_r[k++] = sqrt_w_endpoint * (traj[(T - 1) * n_act + d] - goal[d]);
    }

    return k;
}

__device__ float lst_dot(const float* __restrict__ a, const float* __restrict__ b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

__global__ void ls_trajopt_kernel(
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
    int n_outer_iters,
    int n_ls_iters,
    float lambda_init,
    float w_smooth,
    float w_acc,
    float w_jerk,
    float w_limits,
    float w_trust,
    float w_endpoint,
    float w_collision,
    float w_collision_max,
    float penalty_scale,
    float collision_margin,
    float smooth_min_temperature,
    float max_delta_per_step,
    float fd_eps)
{
    if (threadIdx.x != 0) return;

    int b = blockIdx.x;
    if (b >= B) return;

    int n = T * n_act;
    int m = (5 * T - 3) * n_act + T * LST_G;

    float* base = workspace + (size_t)b * workspace_stride;
    float* qk = base;
    float* dk = qk + n;
    float* Jk = dk + T * LST_G;
    float* r0 = Jk + T * LST_G * n_act;
    float* r1 = r0 + m;
    float* delta = r1 + m;
    float* x_base = delta + n;
    float* T_world = x_base + n;

    float* traj = out_trajs + (size_t)b * n;

    for (int i = 0; i < n; i++) traj[i] = init_trajs[(size_t)b * n + i];
    for (int d = 0; d < n_act; d++) {
        traj[d] = start[d];
        traj[(T - 1) * n_act + d] = goal[d];
    }

    float w_coll = w_collision;
    float lam = lambda_init;

    float sqrt_w_acc = sqrtf(w_smooth * w_acc);
    float sqrt_w_jerk = sqrtf(w_smooth * w_jerk);
    float sqrt_w_trust = sqrtf(w_trust);
    float sqrt_w_limits = sqrtf(w_limits);
    float sqrt_w_endpoint = sqrtf(w_endpoint);

    for (int outer = 0; outer < n_outer_iters; outer++) {
        for (int i = 0; i < n; i++) qk[i] = traj[i];

        for (int t = 0; t < T; t++) {
            float d0[LST_G];
            lst_compute_coll_groups(
                qk + t * n_act,
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
                smooth_min_temperature,
                d0,
                T_world);

            for (int g = 0; g < LST_G; g++) dk[t * LST_G + g] = d0[g];

            float q_tmp[LST_MAX_DOF];
            for (int d = 0; d < n_act; d++) q_tmp[d] = qk[t * n_act + d];

            for (int d = 0; d < n_act; d++) {
                float orig = q_tmp[d];

                q_tmp[d] = orig + fd_eps;
                float dp[LST_G];
                lst_compute_coll_groups(
                    q_tmp,
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
                    smooth_min_temperature,
                    dp,
                    T_world);
                q_tmp[d] = orig;
                float inv_eps = 1.0f / fd_eps;
                for (int g = 0; g < LST_G; g++) {
                    Jk[(t * LST_G + g) * n_act + d] = (dp[g] - d0[g]) * inv_eps;
                }
            }
        }

        for (int inner = 0; inner < n_ls_iters; inner++) {
            float sqrt_w_coll = sqrtf(w_coll);
            int m_used = lst_build_residual(
                traj,
                qk,
                dk,
                Jk,
                lower,
                upper,
                start,
                goal,
                T,
                n_act,
                sqrt_w_acc,
                sqrt_w_jerk,
                sqrt_w_trust,
                sqrt_w_limits,
                sqrt_w_coll,
                sqrt_w_endpoint,
                collision_margin,
                r0);

            float curr_cost = lst_dot(r0, r0, m_used);

            for (int i = 0; i < n; i++) {
                int t = i / n_act;
                if (t == 0 || t == T - 1) {
                    delta[i] = 0.0f;
                    continue;
                }

                float orig = traj[i];
                traj[i] = orig + fd_eps;
                lst_build_residual(
                    traj,
                    qk,
                    dk,
                    Jk,
                    lower,
                    upper,
                    start,
                    goal,
                    T,
                    n_act,
                    sqrt_w_acc,
                    sqrt_w_jerk,
                    sqrt_w_trust,
                    sqrt_w_limits,
                    sqrt_w_coll,
                    sqrt_w_endpoint,
                    collision_margin,
                    r1);

                float rhs = 0.0f;
                float hdiag = 0.0f;
                float inv_eps = 1.0f / fd_eps;
                for (int k = 0; k < m_used; k++) {
                    float dr = (r1[k] - r0[k]) * inv_eps;
                    rhs += dr * r0[k];
                    hdiag += dr * dr;
                }
                rhs = -rhs;
                hdiag += lam;

                float di = rhs / (hdiag + 1e-8f);
                di = fmaxf(-max_delta_per_step, fminf(max_delta_per_step, di));
                delta[i] = di;

                traj[i] = orig;
            }

            for (int i = 0; i < n; i++) x_base[i] = traj[i];

            const float alphas[5] = {1.0f, 0.5f, 0.25f, 0.1f, 0.025f};
            float best_trial = 1e30f;
            float best_alpha = 0.0f;

            for (int ai = 0; ai < 5; ai++) {
                float a = alphas[ai];
                for (int i = 0; i < n; i++) {
                    int d = i % n_act;
                    traj[i] = x_base[i] + a * delta[i];
                    traj[i] = fmaxf(lower[d], fminf(upper[d], traj[i]));
                }
                for (int d = 0; d < n_act; d++) {
                    traj[d] = start[d];
                    traj[(T - 1) * n_act + d] = goal[d];
                }

                lst_build_residual(
                    traj,
                    qk,
                    dk,
                    Jk,
                    lower,
                    upper,
                    start,
                    goal,
                    T,
                    n_act,
                    sqrt_w_acc,
                    sqrt_w_jerk,
                    sqrt_w_trust,
                    sqrt_w_limits,
                    sqrt_w_coll,
                    sqrt_w_endpoint,
                    collision_margin,
                    r1);
                float c = lst_dot(r1, r1, m_used);
                if (c < best_trial) {
                    best_trial = c;
                    best_alpha = a;
                }
            }

            bool improved = best_trial < curr_cost * (1.0f - 1e-4f);
            if (improved) {
                for (int i = 0; i < n; i++) {
                    int d = i % n_act;
                    traj[i] = x_base[i] + best_alpha * delta[i];
                    traj[i] = fmaxf(lower[d], fminf(upper[d], traj[i]));
                }
                for (int d = 0; d < n_act; d++) {
                    traj[d] = start[d];
                    traj[(T - 1) * n_act + d] = goal[d];
                }
                lam = fmaxf(1e-8f, lam * 0.5f);
            } else {
                for (int i = 0; i < n; i++) traj[i] = x_base[i];
                lam = fminf(1e6f, lam * 3.0f);
            }
        }

        w_coll = fminf(w_coll * penalty_scale, w_collision_max);
    }

    float smooth = 0.0f;
    for (int t = 0; t < T - 2; t++) {
        for (int d = 0; d < n_act; d++) {
            float acc = traj[(t + 2) * n_act + d]
                      - 2.0f * traj[(t + 1) * n_act + d]
                      + traj[t * n_act + d];
            smooth += w_smooth * w_acc * acc * acc;
        }
    }
    for (int t = 0; t < T - 3; t++) {
        for (int d = 0; d < n_act; d++) {
            float acc0 = traj[(t + 2) * n_act + d]
                       - 2.0f * traj[(t + 1) * n_act + d]
                       + traj[t * n_act + d];
            float acc1 = traj[(t + 3) * n_act + d]
                       - 2.0f * traj[(t + 2) * n_act + d]
                       + traj[(t + 1) * n_act + d];
            float j = acc1 - acc0;
            smooth += w_smooth * w_jerk * j * j;
        }
    }

    float limits = 0.0f;
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < n_act; d++) {
            float v = fmaxf(0.0f, traj[t * n_act + d] - upper[d])
                    + fmaxf(0.0f, lower[d] - traj[t * n_act + d]);
            limits += w_limits * v * v;
        }
    }

    float coll = 0.0f;
    for (int t = 0; t < T; t++) {
        float dg[LST_G];
        lst_compute_coll_groups(
            traj + t * n_act,
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
            smooth_min_temperature,
            dg,
            T_world);

        for (int g = 0; g < LST_G; g++) {
            float v = fmaxf(0.0f, collision_margin - dg[g]);
            coll += w_collision_max * v * v;
        }
    }

    out_costs[b] = smooth + limits + coll;
}

static ffi::Error LsTrajoptCudaImpl(
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
    int64_t n_ls_iters,
    int64_t S,
    float lambda_init,
    float w_smooth,
    float w_acc,
    float w_jerk,
    float w_limits,
    float w_trust,
    float w_endpoint,
    float w_collision,
    float w_collision_max,
    float penalty_scale,
    float collision_margin,
    float smooth_min_temperature,
    float max_delta_per_step,
    float fd_eps,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_trajs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_costs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_workspace)
{
    int B = static_cast<int>(init_trajs.dimensions()[0]);
    int T = static_cast<int>(init_trajs.dimensions()[1]);
    int n_act = static_cast<int>(init_trajs.dimensions()[2]);
    int n_joints = static_cast<int>(twists.dimensions()[0]);

    int N = static_cast<int>(sphere_offsets.dimensions()[0]) / (static_cast<int>(S) * 3);
    int P = (pair_i_buf.dimensions().size() > 0)
        ? static_cast<int>(pair_i_buf.dimensions()[0])
        : 0;
    int Ms = (world_spheres.dimensions().size() > 0)
        ? static_cast<int>(world_spheres.dimensions()[0])
        : 0;
    int Mc = (world_capsules.dimensions().size() > 0)
        ? static_cast<int>(world_capsules.dimensions()[0])
        : 0;
    int Mb = (world_boxes.dimensions().size() > 0)
        ? static_cast<int>(world_boxes.dimensions()[0])
        : 0;
    int Mh = (world_halfspaces.dimensions().size() > 0)
        ? static_cast<int>(world_halfspaces.dimensions()[0])
        : 0;

    if (n_act > LST_MAX_DOF || n_joints > MAX_JOINTS || P > LST_MAX_PAIRS || static_cast<int>(S) > LST_MAX_S) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "ls_trajopt_cuda dimensions exceed compiled limits");
    }

    int n = T * n_act;
    int m = (5 * T - 3) * n_act + T * LST_G;
    int workspace_stride = n + T * LST_G + T * LST_G * n_act + m + m + n + n + T * n_joints * 7;

    ls_trajopt_kernel<<<B, 1, 0, stream>>>(
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
        static_cast<int>(n_outer_iters),
        static_cast<int>(n_ls_iters),
        lambda_init,
        w_smooth,
        w_acc,
        w_jerk,
        w_limits,
        w_trust,
        w_endpoint,
        w_collision,
        w_collision_max,
        penalty_scale,
        collision_margin,
        smooth_min_temperature,
        max_delta_per_step,
        fd_eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LsTrajoptCudaFfi,
    LsTrajoptCudaImpl,
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
        .Attr<int64_t>("n_outer_iters")
        .Attr<int64_t>("n_ls_iters")
        .Attr<int64_t>("S")
        .Attr<float>("lambda_init")
        .Attr<float>("w_smooth")
        .Attr<float>("w_acc")
        .Attr<float>("w_jerk")
        .Attr<float>("w_limits")
        .Attr<float>("w_trust")
        .Attr<float>("w_endpoint")
        .Attr<float>("w_collision")
        .Attr<float>("w_collision_max")
        .Attr<float>("penalty_scale")
        .Attr<float>("collision_margin")
        .Attr<float>("smooth_min_temperature")
        .Attr<float>("max_delta_per_step")
        .Attr<float>("fd_eps")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
);
