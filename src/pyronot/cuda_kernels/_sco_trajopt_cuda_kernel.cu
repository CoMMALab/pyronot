/**
 * SCO TrajOpt CUDA kernel with XLA FFI binding.
 *
 * Implements the full Sequential Convex Optimization trajectory planner
 * (Schulman et al. 2013) in a single tightly-coupled kernel:
 *
 *   One CUDA thread per trajectory in the batch B.
 *   Each thread runs:
 *     Outer loop (n_outer_iters):
 *       1. FK + finite-difference collision Jacobians for all T waypoints
 *          [d_k: (T, G), J_k: (T, G, n_act)].
 *       2. Inner L-BFGS solve on the convex linearised subproblem
 *          (n_inner_iters steps, no FK in the inner loop).
 *       3. Re-pin endpoints.
 *       4. Scale collision penalty.
 *     Final: nonlinear cost evaluation for best-trajectory selection.
 *
 * G = 5 fixed collision groups (type-based):
 *   Group 0: self-collision  (smooth-min over P link pairs)
 *   Group 1: world spheres   (smooth-min over N × Ms distances)
 *   Group 2: world capsules  (smooth-min over N × Mc distances)
 *   Group 3: world boxes     (smooth-min over N × Mb distances)
 *   Group 4: world halfspaces(smooth-min over N × Mh distances)
 *
 * Reuses:
 *   _ik_cuda_helpers.cuh  — fk_single(), lbfgs_two_loop(), math helpers
 *
 * Build with:
 *   bash src/pyronot/cuda_kernels/build_sco_trajopt_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits
// ---------------------------------------------------------------------------

#ifndef SCO_MAX_T
#define SCO_MAX_T     30
#endif

#ifndef SCO_MAX_DOF
#define SCO_MAX_DOF   MAX_ACT   // 16, from _ik_cuda_helpers.cuh
#endif

#ifndef SCO_MAX_N
#define SCO_MAX_N     MAX_JOINTS  // 64
#endif

#ifndef SCO_MAX_S
#define SCO_MAX_S     8    // max spheres per link
#endif

#define SCO_MAX_G     5    // fixed: 1 self + 4 world types

#ifndef SCO_MAX_M
#define SCO_MAX_M     8    // max L-BFGS history pairs
#endif

#ifndef SCO_MAX_PAIRS
#define SCO_MAX_PAIRS 256  // max self-collision link pairs
#endif

// Derived sizes
#define SCO_N_TRAJ   (SCO_MAX_T * SCO_MAX_DOF)   // 480 — flattened traj size
#define SCO_N_JK     (SCO_MAX_T * SCO_MAX_G * SCO_MAX_DOF) // 2400 — J_k size

// ---------------------------------------------------------------------------
// Collision distance primitives (device-only, duplicated for self-containment)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float sco_sphere_sphere_dist(
    float ax, float ay, float az, float ar,
    float bx, float by, float bz, float br)
{
    float dx = ax-bx, dy = ay-by, dz = az-bz;
    return sqrtf(dx*dx + dy*dy + dz*dz) - (ar + br);
}

__device__ __forceinline__ float sco_sphere_capsule_dist(
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

__device__ __forceinline__ float sco_box_sdf_local(
    float p1, float p2, float p3,
    float hl1, float hl2, float hl3)
{
    float q1 = fabsf(p1)-hl1, q2 = fabsf(p2)-hl2, q3 = fabsf(p3)-hl3;
    float mq1 = fmaxf(q1,0.f), mq2 = fmaxf(q2,0.f), mq3 = fmaxf(q3,0.f);
    return sqrtf(mq1*mq1 + mq2*mq2 + mq3*mq3)
           + fminf(fmaxf(fmaxf(q1,q2),q3), 0.f);
}

__device__ __forceinline__ float sco_sphere_box_dist(
    float sx, float sy, float sz, float sr,
    float bcx, float bcy, float bcz,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float a3x, float a3y, float a3z,
    float hl1, float hl2, float hl3)
{
    float dx = sx-bcx, dy = sy-bcy, dz = sz-bcz;
    float p1 = dx*a1x + dy*a1y + dz*a1z;
    float p2 = dx*a2x + dy*a2y + dz*a2z;
    float p3 = dx*a3x + dy*a3y + dz*a3z;
    return sco_box_sdf_local(p1, p2, p3, hl1, hl2, hl3) - sr;
}

__device__ __forceinline__ float sco_sphere_halfspace_dist(
    float sx, float sy, float sz, float sr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    return (sx-px)*nx + (sy-py)*ny + (sz-pz)*nz - sr;
}

// ---------------------------------------------------------------------------
// Apply SE(3) transform to a point
// ---------------------------------------------------------------------------

__device__ __forceinline__ void sco_apply_se3(
    const float* __restrict__ T,   // [7]: [w,x,y,z,tx,ty,tz]
    const float* __restrict__ p,   // [3]: local point
    float* __restrict__ out)       // [3]: world point
{
    quat_rotate(T, p, out);
    out[0] += T[4];
    out[1] += T[5];
    out[2] += T[6];
}

// ---------------------------------------------------------------------------
// colldist_from_sdf (matches Python collision/_collision.py)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float sco_colldist_from_sdf(float d, float margin)
{
    d = fminf(d, margin);
    float val;
    if (d < 0.0f) {
        val = d - 0.5f * margin;
    } else {
        float diff = d - margin;
        val = -0.5f / (margin + 1e-6f) * diff * diff;
    }
    return fminf(val, 0.0f);
}

// ---------------------------------------------------------------------------
// Incremental logsumexp accumulator for smooth-min
//   smooth_min(d) = -tau * logsumexp(-d / tau)
// ---------------------------------------------------------------------------

struct SmoothMinAcc {
    float max_val;  // running max of (-d / tau)
    float sum_exp;  // running sum of exp((-d/tau) - max_val)

    __device__ void init() { max_val = -1e30f; sum_exp = 0.0f; }

    __device__ void update(float d, float temperature) {
        float v = -d / temperature;
        if (v > max_val) {
            sum_exp *= expf(max_val - v);
            max_val = v;
        }
        sum_exp += expf(v - max_val);
    }

    __device__ float finalize(float temperature) const {
        if (sum_exp <= 0.0f) return 1e10f;   // no obstacles → very large
        return -temperature * (logf(sum_exp) + max_val);
    }
};

// ---------------------------------------------------------------------------
// Compute G collision distances for one configuration
// Output: dists[G] using incremental smooth-min accumulators
// ---------------------------------------------------------------------------

__device__ void sco_compute_coll_dists(
    const float* __restrict__ cfg,          // [n_act]
    // FK shared params
    const float* __restrict__ s_twists,
    const float* __restrict__ s_parent_tf,
    const int*   __restrict__ s_parent_idx,
    const int*   __restrict__ s_act_idx,
    const float* __restrict__ s_mimic_mul,
    const float* __restrict__ s_mimic_off,
    const int*   __restrict__ s_mimic_act_idx,
    const int*   __restrict__ s_topo_inv,
    // Robot collision params (shared)
    const float* __restrict__ s_sphere_off, // [N*S*3]
    const float* __restrict__ s_sphere_rad, // [N*S]
    const int*   __restrict__ s_pair_i,     // [P]
    const int*   __restrict__ s_pair_j,     // [P]
    // World geometry (global memory)
    const float* __restrict__ world_spheres,    // [Ms, 4]
    const float* __restrict__ world_capsules,   // [Mc, 7]
    const float* __restrict__ world_boxes,      // [Mb, 15]
    const float* __restrict__ world_halfspaces, // [Mh, 6]
    // Dimensions
    int n_joints, int n_act, int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    float temperature,
    // Output
    float* __restrict__ dists,   // [G=5]
    float* __restrict__ T_world) // [n_joints*7] workspace
{
    // Step 1: FK
    fk_single(cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
              s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
              T_world, n_joints, n_act);

    // Step 2: Collision distances with incremental smooth-min
    SmoothMinAcc acc[SCO_MAX_G];
    for (int g = 0; g < SCO_MAX_G; g++) acc[g].init();

    // --- Group 0: self-collision (link pairs) ---
    for (int p = 0; p < P; p++) {
        int li = s_pair_i[p];
        int lj = s_pair_j[p];
        float min_d = 1e10f;

        for (int si = 0; si < S; si++) {
            float ri = s_sphere_rad[li * S + si];
            if (ri < 0.0f) continue;
            float local_ci[3] = {
                s_sphere_off[(li*S+si)*3+0],
                s_sphere_off[(li*S+si)*3+1],
                s_sphere_off[(li*S+si)*3+2]
            };
            float ci[3];
            sco_apply_se3(T_world + li*7, local_ci, ci);

            for (int sj = 0; sj < S; sj++) {
                float rj = s_sphere_rad[lj * S + sj];
                if (rj < 0.0f) continue;
                float local_cj[3] = {
                    s_sphere_off[(lj*S+sj)*3+0],
                    s_sphere_off[(lj*S+sj)*3+1],
                    s_sphere_off[(lj*S+sj)*3+2]
                };
                float cj[3];
                sco_apply_se3(T_world + lj*7, local_cj, cj);
                float d = sco_sphere_sphere_dist(
                    ci[0], ci[1], ci[2], ri,
                    cj[0], cj[1], cj[2], rj);
                if (d < min_d) min_d = d;
            }
        }
        if (min_d < 1e9f)
            acc[0].update(min_d, temperature);
    }

    // --- Groups 1-4: world obstacles ---
    for (int j = 0; j < N; j++) {
        for (int s = 0; s < S; s++) {
            float r = s_sphere_rad[j * S + s];
            if (r < 0.0f) continue;
            float local_c[3] = {
                s_sphere_off[(j*S+s)*3+0],
                s_sphere_off[(j*S+s)*3+1],
                s_sphere_off[(j*S+s)*3+2]
            };
            float wc[3];
            sco_apply_se3(T_world + j*7, local_c, wc);

            // Group 1: world spheres
            for (int m = 0; m < Ms; m++) {
                float d = sco_sphere_sphere_dist(
                    wc[0], wc[1], wc[2], r,
                    world_spheres[m*4+0], world_spheres[m*4+1],
                    world_spheres[m*4+2], world_spheres[m*4+3]);
                acc[1].update(d, temperature);
            }

            // Group 2: world capsules
            for (int m = 0; m < Mc; m++) {
                const float* cap = world_capsules + m*7;
                float d = sco_sphere_capsule_dist(
                    wc[0], wc[1], wc[2], r,
                    cap[0], cap[1], cap[2],
                    cap[3], cap[4], cap[5], cap[6]);
                acc[2].update(d, temperature);
            }

            // Group 3: world boxes
            for (int m = 0; m < Mb; m++) {
                const float* box = world_boxes + m*15;
                float d = sco_sphere_box_dist(
                    wc[0], wc[1], wc[2], r,
                    box[0],  box[1],  box[2],   // center
                    box[3],  box[4],  box[5],   // axis 1
                    box[6],  box[7],  box[8],   // axis 2
                    box[9],  box[10], box[11],  // axis 3
                    box[12], box[13], box[14]); // half-lengths
                acc[3].update(d, temperature);
            }

            // Group 4: world halfspaces
            for (int m = 0; m < Mh; m++) {
                const float* hs = world_halfspaces + m*6;
                float d = sco_sphere_halfspace_dist(
                    wc[0], wc[1], wc[2], r,
                    hs[0], hs[1], hs[2],
                    hs[3], hs[4], hs[5]);
                acc[4].update(d, temperature);
            }
        }
    }

    for (int g = 0; g < SCO_MAX_G; g++)
        dists[g] = acc[g].finalize(temperature);
}

// ---------------------------------------------------------------------------
// Compute d_k[T, G] and J_k[T, G, n_act] via central finite differences
// ---------------------------------------------------------------------------

__device__ void sco_compute_coll_jacs(
    const float* __restrict__ traj,         // [T, n_act]
    int T, int n_act, int n_joints,
    int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    float temperature, float fd_eps,
    // FK shared params
    const float* __restrict__ s_twists,
    const float* __restrict__ s_parent_tf,
    const int*   __restrict__ s_parent_idx,
    const int*   __restrict__ s_act_idx,
    const float* __restrict__ s_mimic_mul,
    const float* __restrict__ s_mimic_off,
    const int*   __restrict__ s_mimic_act_idx,
    const int*   __restrict__ s_topo_inv,
    // Robot collision shared params
    const float* __restrict__ s_sphere_off,
    const float* __restrict__ s_sphere_rad,
    const int*   __restrict__ s_pair_i,
    const int*   __restrict__ s_pair_j,
    // World geometry
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    // Output
    float* __restrict__ d_k,     // [T * G]
    float* __restrict__ J_k,     // [T * G * n_act]
    // Workspace (thread-local)
    float* __restrict__ T_world, // [n_joints * 7]
    float* __restrict__ q_perturb) // [n_act]
{
    for (int t = 0; t < T; t++) {
        const float* q_t = traj + t * n_act;

        // Base distances at q_t
        float d_base[SCO_MAX_G];
        sco_compute_coll_dists(
            q_t,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_sphere_off, s_sphere_rad, s_pair_i, s_pair_j,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
            n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
            temperature, d_base, T_world);

        // Store d_k[t]
        for (int g = 0; g < SCO_MAX_G; g++)
            d_k[t * SCO_MAX_G + g] = d_base[g];

        // Finite-difference Jacobian: J_k[t, g, d] via central differences
        for (int d = 0; d < n_act; d++) {
            // Copy q_t and apply +/- perturbation
            for (int a = 0; a < n_act; a++) q_perturb[a] = q_t[a];

            float d_plus[SCO_MAX_G], d_minus[SCO_MAX_G];

            q_perturb[d] = q_t[d] + fd_eps;
            sco_compute_coll_dists(
                q_perturb,
                s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                s_sphere_off, s_sphere_rad, s_pair_i, s_pair_j,
                world_spheres, world_capsules, world_boxes, world_halfspaces,
                n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
                temperature, d_plus, T_world);

            q_perturb[d] = q_t[d] - fd_eps;
            sco_compute_coll_dists(
                q_perturb,
                s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                s_sphere_off, s_sphere_rad, s_pair_i, s_pair_j,
                world_spheres, world_capsules, world_boxes, world_halfspaces,
                n_joints, n_act, N, S, P, Ms, Mc, Mb, Mh,
                temperature, d_minus, T_world);

            float inv_2eps = 1.0f / (2.0f * fd_eps);
            for (int g = 0; g < SCO_MAX_G; g++) {
                J_k[(t * SCO_MAX_G + g) * n_act + d] =
                    (d_plus[g] - d_minus[g]) * inv_2eps;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Smoothness cost and gradient (analytic, 4th-order central-difference)
// ---------------------------------------------------------------------------

__device__ void sco_smoothness_cost_grad(
    const float* __restrict__ q,  // [T * n_act]
    int T, int n_act,
    float w_acc, float w_jerk,
    float* __restrict__ cost,
    float* __restrict__ grad)     // [T * n_act], accumulated (not zeroed here)
{
    const float stencil[5] = {
        -1.0f/12.0f, 16.0f/12.0f, -30.0f/12.0f, 16.0f/12.0f, -1.0f/12.0f
    };

    // Compute acceleration: acc[t] for t = 0..T-5
    float acc[SCO_MAX_T * SCO_MAX_DOF];
    float jerk[SCO_MAX_T * SCO_MAX_DOF];

    *cost = 0.0f;

    int n_acc  = (T >= 5) ? (T - 4) : 0;
    int n_jerk = (T >= 6) ? (T - 5) : 0;

    for (int t = 0; t < n_acc; t++) {
        for (int d = 0; d < n_act; d++) {
            float a = 0.0f;
            for (int k = 0; k < 5; k++)
                a += stencil[k] * q[(t + k) * n_act + d];
            acc[t * n_act + d] = a;
            *cost += w_acc * a * a;
        }
    }

    for (int t = 0; t < n_jerk; t++) {
        for (int d = 0; d < n_act; d++) {
            float j = acc[(t+1)*n_act + d] - acc[t*n_act + d];
            jerk[t * n_act + d] = j;
            *cost += w_jerk * j * j;
        }
    }

    // Gradient: accumulate into grad[]
    for (int s = 0; s < T; s++) {
        for (int d = 0; d < n_act; d++) {
            float g = 0.0f;

            // Acceleration term
            int t_lo = (s - 4 > 0) ? s - 4 : 0;
            int t_hi = (s < n_acc) ? s : n_acc - 1;
            for (int t = t_lo; t <= t_hi; t++) {
                int k = s - t;   // 0 <= k <= 4
                g += 2.0f * w_acc * acc[t*n_act + d] * stencil[k];
            }

            // Jerk term: jerk[t] = acc[t+1] - acc[t], t = 0..n_jerk-1
            // d(jerk[t])/d(q[s]) = stencil[s-t-1] - stencil[s-t]
            int tj_lo = (s - 5 > 0) ? s - 5 : 0;
            int tj_hi = (s < n_jerk) ? s : n_jerk - 1;
            for (int t = tj_lo; t <= tj_hi; t++) {
                int k1 = s - t - 1;
                int k2 = s - t;
                float c1 = (k1 >= 0 && k1 <= 4) ? stencil[k1] : 0.0f;
                float c2 = (k2 >= 0 && k2 <= 4) ? stencil[k2] : 0.0f;
                g += 2.0f * w_jerk * jerk[t*n_act + d] * (c1 - c2);
            }

            grad[s * n_act + d] += g;
        }
    }
}

// ---------------------------------------------------------------------------
// Joint-limit cost and gradient (soft squared exceedance)
// ---------------------------------------------------------------------------

__device__ void sco_limits_cost_grad(
    const float* __restrict__ q,      // [T * n_act]
    int T, int n_act,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    float w_limits,
    float* __restrict__ cost,
    float* __restrict__ grad)          // [T * n_act], accumulated
{
    *cost = 0.0f;
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < n_act; d++) {
            float qv   = q[t * n_act + d];
            float viol = fmaxf(0.0f, qv - upper[d])
                       + fmaxf(0.0f, lower[d] - qv);
            *cost += w_limits * viol * viol;

            float gv  = 2.0f * w_limits * viol;
            float sign = (qv > upper[d]) ? 1.0f
                       : (qv < lower[d]) ? -1.0f : 0.0f;
            grad[t * n_act + d] += gv * sign;
        }
    }
}

// ---------------------------------------------------------------------------
// Inner linearised cost + gradient (no FK needed)
// ---------------------------------------------------------------------------

__device__ void sco_inner_cost_grad(
    const float* __restrict__ x,   // current iterate [T * n_act]
    const float* __restrict__ q_k, // linearisation point [T * n_act]
    const float* __restrict__ d_k, // distances at q_k [T * G]
    const float* __restrict__ J_k, // Jacobians at q_k [T * G * n_act]
    int T, int n_act,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_trust, float w_coll,
    float collision_margin,
    float* __restrict__ cost,
    float* __restrict__ grad)  // [T * n_act] — output (zeroed inside)
{
    int n = T * n_act;
    for (int i = 0; i < n; i++) grad[i] = 0.0f;
    *cost = 0.0f;

    // Smoothness
    float c_smooth;
    sco_smoothness_cost_grad(x, T, n_act, w_acc, w_jerk, &c_smooth, grad);
    *cost += w_smooth * c_smooth;

    // Limits
    float c_limits;
    sco_limits_cost_grad(x, T, n_act, lower, upper, w_limits, &c_limits, grad);
    *cost += c_limits;

    // Linearised collision cost: w_coll * sum_{t,g} max(0, margin - d_lin)^2
    // d_lin[t,g] = d_k[t,g] + J_k[t,g,:] . (x[t] - q_k[t])
    for (int t = 0; t < T; t++) {
        for (int g = 0; g < SCO_MAX_G; g++) {
            float dk_tg = d_k[t * SCO_MAX_G + g];
            // delta_q = x[t] - q_k[t]
            float d_lin = dk_tg;
            for (int d = 0; d < n_act; d++) {
                float delta = x[t*n_act+d] - q_k[t*n_act+d];
                d_lin += J_k[(t*SCO_MAX_G+g)*n_act + d] * delta;
            }
            float viol = fmaxf(0.0f, collision_margin - d_lin);
            *cost += w_coll * viol * viol;
            // Gradient: -2 * w_coll * viol * J_k[t,g,:]
            float gcoeff = -2.0f * w_coll * viol;
            for (int d = 0; d < n_act; d++) {
                grad[t*n_act + d] +=
                    gcoeff * J_k[(t*SCO_MAX_G+g)*n_act + d];
            }
        }
    }

    // Trust region: w_trust * ||x - q_k||^2
    for (int i = 0; i < n; i++) {
        float delta = x[i] - q_k[i];
        *cost += w_trust * delta * delta;
        grad[i] += 2.0f * w_trust * delta;
    }
}

// ---------------------------------------------------------------------------
// Inner L-BFGS solve (convex linearised subproblem)
// ---------------------------------------------------------------------------

__device__ void sco_lbfgs_inner_solve(
    float* __restrict__ traj,       // [T * n_act] in/out
    const float* __restrict__ q_k,  // [T * n_act] linearisation point
    const float* __restrict__ d_k,  // [T * G]
    const float* __restrict__ J_k,  // [T * G * n_act]
    int T, int n_act,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_trust, float w_coll,
    float collision_margin,
    int n_inner_iters, int m_lbfgs,
    // Workspace scratch buffers (all from global workspace, no local arrays)
    float* __restrict__ s_buf,      // [MAX_M * T * n_act]
    float* __restrict__ y_buf,      // [MAX_M * T * n_act]
    float* __restrict__ rho_buf,    // [MAX_M]
    float* __restrict__ alpha_buf,  // [MAX_M]
    float* __restrict__ g,          // [T * n_act] — current gradient
    float* __restrict__ g_prev,     // [T * n_act] — previous gradient
    float* __restrict__ direction,  // [T * n_act]
    float* __restrict__ best_x,     // [T * n_act]
    float* __restrict__ cur_g,      // [T * n_act] — scratch for inner gradient
    float* __restrict__ x_trial,    // [T * n_act] — scratch for line search
    float* __restrict__ g_trial)    // [T * n_act] — scratch for line search
{
    int n = T * n_act;
    int m = m_lbfgs;  // could be < SCO_MAX_M

    // Endpoint mask: zero gradient at t=0 and t=T-1 (pin start and goal)
    // Applied inline when storing/applying directions.

    // Initial cost and gradient
    float best_cost;
    sco_inner_cost_grad(
        traj, q_k, d_k, J_k, T, n_act,
        lower, upper,
        w_smooth, w_acc, w_jerk, w_limits, w_trust, w_coll,
        collision_margin, &best_cost, g);

    // Pin endpoints in gradient
    for (int d = 0; d < n_act; d++) {
        g[d] = 0.0f;
        g[(T-1)*n_act + d] = 0.0f;
    }

    // Save best iterate
    for (int i = 0; i < n; i++) best_x[i] = traj[i];

    // L-BFGS state
    int m_used = 0;
    int newest = 0;  // index of most recently stored pair (wraps)
    int iter_count = 0;

    for (int iter = 0; iter < n_inner_iters; iter++) {
        float cost_val;
        sco_inner_cost_grad(
            traj, q_k, d_k, J_k, T, n_act,
            lower, upper,
            w_smooth, w_acc, w_jerk, w_limits, w_trust, w_coll,
            collision_margin, &cost_val, cur_g);

        // Pin endpoints
        for (int d = 0; d < n_act; d++) {
            cur_g[d] = 0.0f;
            cur_g[(T-1)*n_act + d] = 0.0f;
        }

        // Update L-BFGS curvature history (skip first iteration)
        if (iter_count > 0) {
            float sy = 0.0f, yy = 0.0f;
            for (int i = 0; i < n; i++) {
                float si = traj[i] - (traj[i] - direction[i] /* x_prev ≈ traj - last_step */);
                // Use g_prev to form y = g - g_prev
                float yi_val = cur_g[i] - g_prev[i];
                sy += direction[i] * yi_val;  // direction was s_k = x_new - x_prev
                yy += yi_val * yi_val;
            }
            // direction still holds s_k = x - x_prev from last iteration

            bool valid = (sy > 1e-10f * yy + 1e-30f);
            if (valid) {
                int new_newest = (newest + 1) % m;
                for (int i = 0; i < n; i++) {
                    s_buf[new_newest * n + i] = direction[i];  // s_k
                    float yi_val = cur_g[i] - g_prev[i];
                    y_buf[new_newest * n + i] = yi_val;
                }
                rho_buf[new_newest] = 1.0f / (sy + 1e-30f);
                newest = new_newest;
                if (m_used < m) m_used++;
            }
        }

        // Save g_prev = cur_g, direction = step (for next curvature update)
        for (int i = 0; i < n; i++) g_prev[i] = cur_g[i];

        // Compute L-BFGS direction
        lbfgs_two_loop(cur_g, s_buf, y_buf, rho_buf, alpha_buf,
                       n, m, m_used, newest, direction);
        // direction now holds -H*g (descent direction)

        // Pin endpoints in direction
        for (int d = 0; d < n_act; d++) {
            direction[d] = 0.0f;
            direction[(T-1)*n_act + d] = 0.0f;
        }

        // Line search over 5 alpha candidates
        const float alphas[5] = {1.0f, 0.5f, 0.25f, 0.1f, 0.025f};
        float best_alpha_cost = 1e30f;
        int best_alpha_idx = 0;
        float suff_thresh = cost_val * (1.0f - 1e-4f);

        for (int ai = 0; ai < 5; ai++) {
            for (int i = 0; i < n; i++)
                x_trial[i] = traj[i] + alphas[ai] * direction[i];

            float c_trial;
            sco_inner_cost_grad(
                x_trial, q_k, d_k, J_k, T, n_act,
                lower, upper,
                w_smooth, w_acc, w_jerk, w_limits, w_trust, w_coll,
                collision_margin, &c_trial, g_trial);

            if (c_trial < best_alpha_cost) {
                best_alpha_cost = c_trial;
                best_alpha_idx  = ai;
            }
            if (c_trial < suff_thresh) break;  // early exit
        }

        // Update x
        float alpha_star = alphas[best_alpha_idx];
        // Save x_prev in direction (temporarily) before updating traj
        for (int i = 0; i < n; i++) {
            float x_prev_i = traj[i];
            traj[i] += alpha_star * direction[i];
            direction[i] = traj[i] - x_prev_i;  // now direction = s_k for next iter
        }

        if (best_alpha_cost < best_cost) {
            best_cost = best_alpha_cost;
            for (int i = 0; i < n; i++) best_x[i] = traj[i];
        }

        iter_count++;
    }

    // Return best iterate
    for (int i = 0; i < n; i++) traj[i] = best_x[i];
}

// ---------------------------------------------------------------------------
// Nonlinear cost evaluation (for final ranking, uses actual FK+collision)
// ---------------------------------------------------------------------------

__device__ float sco_eval_nonlinear_cost(
    const float* __restrict__ traj,  // [T * n_act]
    int T, int n_act, int n_joints,
    int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_coll_max,
    float collision_margin,
    // FK shared params
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
    float* __restrict__ T_world,   // [n_joints*7] workspace
    float* __restrict__ g_dummy)   // [T*n_act] scratch for cost/grad calls
{
    float cost = 0.0f;

    // Smoothness and limits cost
    for (int i = 0; i < T*n_act; i++) g_dummy[i] = 0.0f;
    float c_smooth, c_limits;
    sco_smoothness_cost_grad(traj, T, n_act, w_acc, w_jerk, &c_smooth, g_dummy);
    sco_limits_cost_grad(traj, T, n_act, lower, upper, w_limits, &c_limits, g_dummy);
    cost += w_smooth * c_smooth + c_limits;

    // Nonlinear collision cost (actual FK, all individual distances)
    for (int t = 0; t < T; t++) {
        const float* q_t = traj + t * n_act;
        fk_single(q_t, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                  s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                  T_world, n_joints, n_act);

        // Self-collision pairs
        for (int p = 0; p < P; p++) {
            int li = s_pair_i[p], lj = s_pair_j[p];
            float min_d = 1e10f;
            for (int si = 0; si < S; si++) {
                float ri = s_sphere_rad[li*S + si];
                if (ri < 0.0f) continue;
                float lci[3] = {
                    s_sphere_off[(li*S+si)*3+0],
                    s_sphere_off[(li*S+si)*3+1],
                    s_sphere_off[(li*S+si)*3+2] };
                float ci[3]; sco_apply_se3(T_world + li*7, lci, ci);
                for (int sj = 0; sj < S; sj++) {
                    float rj = s_sphere_rad[lj*S + sj];
                    if (rj < 0.0f) continue;
                    float lcj[3] = {
                        s_sphere_off[(lj*S+sj)*3+0],
                        s_sphere_off[(lj*S+sj)*3+1],
                        s_sphere_off[(lj*S+sj)*3+2] };
                    float cj[3]; sco_apply_se3(T_world + lj*7, lcj, cj);
                    float d = sco_sphere_sphere_dist(
                        ci[0],ci[1],ci[2],ri,cj[0],cj[1],cj[2],rj);
                    if (d < min_d) min_d = d;
                }
            }
            if (min_d < 1e9f)
                cost -= fminf(sco_colldist_from_sdf(min_d, collision_margin), 0.0f)
                        * w_coll_max;
        }

        // World collision
        for (int j = 0; j < N; j++) {
            for (int s = 0; s < S; s++) {
                float r = s_sphere_rad[j*S + s];
                if (r < 0.0f) continue;
                float lc[3] = {
                    s_sphere_off[(j*S+s)*3+0],
                    s_sphere_off[(j*S+s)*3+1],
                    s_sphere_off[(j*S+s)*3+2] };
                float wc[3]; sco_apply_se3(T_world + j*7, lc, wc);

                for (int m = 0; m < Ms; m++) {
                    float d = sco_sphere_sphere_dist(
                        wc[0],wc[1],wc[2],r,
                        world_spheres[m*4+0], world_spheres[m*4+1],
                        world_spheres[m*4+2], world_spheres[m*4+3]);
                    cost -= fminf(sco_colldist_from_sdf(d, collision_margin),0.f)
                            * w_coll_max;
                }
                for (int m = 0; m < Mc; m++) {
                    const float* cap = world_capsules + m*7;
                    float d = sco_sphere_capsule_dist(
                        wc[0],wc[1],wc[2],r,
                        cap[0],cap[1],cap[2],cap[3],cap[4],cap[5],cap[6]);
                    cost -= fminf(sco_colldist_from_sdf(d, collision_margin),0.f)
                            * w_coll_max;
                }
                for (int m = 0; m < Mb; m++) {
                    const float* box = world_boxes + m*15;
                    float d = sco_sphere_box_dist(
                        wc[0],wc[1],wc[2],r,
                        box[0],box[1],box[2],
                        box[3],box[4],box[5],
                        box[6],box[7],box[8],
                        box[9],box[10],box[11],
                        box[12],box[13],box[14]);
                    cost -= fminf(sco_colldist_from_sdf(d, collision_margin),0.f)
                            * w_coll_max;
                }
                for (int m = 0; m < Mh; m++) {
                    const float* hs = world_halfspaces + m*6;
                    float d = sco_sphere_halfspace_dist(
                        wc[0],wc[1],wc[2],r,
                        hs[0],hs[1],hs[2],hs[3],hs[4],hs[5]);
                    cost -= fminf(sco_colldist_from_sdf(d, collision_margin),0.f)
                            * w_coll_max;
                }
            }
        }
    }
    return cost;
}

// ---------------------------------------------------------------------------
// Main SCO TrajOpt kernel — one thread per trajectory
// ---------------------------------------------------------------------------

/**
 * @param init_trajs      [B, T, n_act]        initial trajectories
 * @param twists          [n_joints, 6]         FK twist params
 * @param parent_tf       [n_joints, 7]         FK parent transforms
 * @param parent_idx      [n_joints]
 * @param act_idx         [n_joints]
 * @param mimic_mul       [n_joints]
 * @param mimic_off       [n_joints]
 * @param mimic_act_idx   [n_joints]
 * @param topo_inv        [n_joints]
 * @param sphere_offsets  [N*S*3]  sphere centers in link frame (row-major [N,S,3])
 * @param sphere_radii    [N*S]    sphere radii (negative = padding)
 * @param pair_i          [P]      self-collision link pairs, first index
 * @param pair_j          [P]      self-collision link pairs, second index
 * @param world_spheres   [Ms, 4]
 * @param world_capsules  [Mc, 7]
 * @param world_boxes     [Mb, 15]
 * @param world_halfspaces[Mh, 6]
 * @param lower           [n_act]  joint lower limits
 * @param upper           [n_act]  joint upper limits
 * @param start           [n_act]  start configuration (pinned)
 * @param goal            [n_act]  goal configuration (pinned)
 * @param out_trajs       [B, T, n_act]  output trajectories
 * @param out_costs       [B]            final nonlinear costs
 */
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
    float*       __restrict__ workspace,    // [B * workspace_stride] scratch
    int workspace_stride,
    int B, int T, int n_joints, int n_act,
    int N, int S, int P,
    int Ms, int Mc, int Mb, int Mh,
    int   n_outer_iters, int n_inner_iters, int m_lbfgs,
    float w_smooth, float w_acc, float w_jerk,
    float w_limits, float w_trust,
    float w_collision, float w_collision_max, float penalty_scale,
    float collision_margin, float smooth_min_temperature, float fd_eps)
{
    // ── Shared memory: robot params loaded once per block ─────────────────
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

    // Cooperative load into shared memory
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
    int ns3 = N * S * 3;
    int ns  = N * S;
    for (int i = threadIdx.x; i < ns3; i += blockDim.x) s_sphere_off[i] = sphere_offsets[i];
    for (int i = threadIdx.x; i < ns;  i += blockDim.x) s_sphere_rad[i] = sphere_radii[i];
    for (int i = threadIdx.x; i < P;   i += blockDim.x) {
        s_pair_i[i] = pair_i[i];
        s_pair_j[i] = pair_j[i];
    }
    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i] = lower[i];
        s_upper[i] = upper[i];
        s_start[i] = start[i];
        s_goal[i]  = goal[i];
    }
    __syncthreads();

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // ── Thread-private state: pointers into global workspace ──────────────
    // Layout (floats, per thread b):
    //   [0..n)            traj        [T * n_act]
    //   [n..2n)           q_k         [T * n_act]
    //   [2n..3n)          g           [T * n_act]
    //   [3n..4n)          g_prev      [T * n_act]
    //   [4n..5n)          dir_buf     [T * n_act]
    //   [5n..6n)          best_x      [T * n_act]
    //   [6n..7n)          cur_g       [T * n_act]  (lbfgs inner scratch)
    //   [7n..8n)          x_trial     [T * n_act]  (line search scratch)
    //   [8n..9n)          g_trial     [T * n_act]  (line search scratch)
    //   [9n..10n)         g_dummy     [T * n_act]  (eval_nonlinear_cost scratch)
    //   [10n..10n+T*G)    d_k         [T * G]
    //   [10n+T*G..)       J_k         [T * G * n_act]
    //   ..                s_buf       [m * n]
    //   ..                y_buf       [m * n]
    //   ..                rho_buf     [SCO_MAX_M]
    //   ..                alpha_b     [SCO_MAX_M]
    //   ..                T_world     [SCO_MAX_N * 7]
    //   ..                q_tmp       [n_act]
    const int n   = T * n_act;
    const int TG  = T * SCO_MAX_G;
    float* base   = workspace + (size_t)b * workspace_stride;
    float* traj    = base;
    float* q_k     = base +  n;
    float* g       = base + 2*n;
    float* g_prev  = base + 3*n;
    float* dir_buf = base + 4*n;
    float* best_x  = base + 5*n;
    float* cur_g   = base + 6*n;
    float* x_trial = base + 7*n;
    float* g_trial = base + 8*n;
    float* g_dummy = base + 9*n;
    float* d_k     = base + 10*n;
    float* J_k     = d_k  + TG;
    float* s_buf   = J_k  + TG * n_act;
    float* y_buf   = s_buf + m_lbfgs * n;
    float* rho_buf = y_buf + m_lbfgs * n;
    float* alpha_b = rho_buf + SCO_MAX_M;
    float* T_world = alpha_b + SCO_MAX_M;
    float* q_tmp   = T_world + SCO_MAX_N * 7;

    // Load initial trajectory
    int stride = T * n_act;
    for (int i = 0; i < stride; i++)
        traj[i] = init_trajs[b * stride + i];

    // Pin endpoints
    for (int d = 0; d < n_act; d++) {
        traj[d]               = s_start[d];
        traj[(T-1)*n_act + d] = s_goal[d];
    }

    // Outer SCO loop
    float w_coll = w_collision;

    for (int outer = 0; outer < n_outer_iters; outer++) {
        // Step 1: Copy current trajectory as linearisation point
        for (int i = 0; i < stride; i++) q_k[i] = traj[i];

        // Step 2: Compute d_k and J_k
        sco_compute_coll_jacs(
            traj, T, n_act, n_joints, N, S, P, Ms, Mc, Mb, Mh,
            smooth_min_temperature, fd_eps,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_sphere_off, s_sphere_rad, s_pair_i, s_pair_j,
            world_spheres, world_capsules, world_boxes, world_halfspaces,
            d_k, J_k, T_world, q_tmp);

        // Step 3: Inner L-BFGS solve
        sco_lbfgs_inner_solve(
            traj, q_k, d_k, J_k, T, n_act,
            s_lower, s_upper,
            w_smooth, w_acc, w_jerk, w_limits, w_trust, w_coll,
            collision_margin, n_inner_iters, m_lbfgs,
            s_buf, y_buf, rho_buf, alpha_b,
            g, g_prev, dir_buf, best_x,
            cur_g, x_trial, g_trial);

        // Step 4: Re-pin endpoints
        for (int d = 0; d < n_act; d++) {
            traj[d]               = s_start[d];
            traj[(T-1)*n_act + d] = s_goal[d];
        }

        // Step 5: Scale collision penalty
        w_coll = fminf(w_coll * penalty_scale, w_collision_max);
    }

    // Final nonlinear cost for ranking
    float final_cost = sco_eval_nonlinear_cost(
        traj, T, n_act, n_joints, N, S, P, Ms, Mc, Mb, Mh,
        s_lower, s_upper,
        w_smooth, w_acc, w_jerk, w_limits, w_collision_max,
        collision_margin,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_sphere_off, s_sphere_rad, s_pair_i, s_pair_j,
        world_spheres, world_capsules, world_boxes, world_halfspaces,
        T_world, g_dummy);

    // Write outputs
    for (int i = 0; i < stride; i++)
        out_trajs[b * stride + i] = traj[i];
    out_costs[b] = final_cost;
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

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
    int64_t S,     // spheres per link
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
    const int N        = static_cast<int>(sphere_offsets.dimensions()[0]) / (static_cast<int>(S) * 3);
    const int P        = (pair_i_buf.dimensions().size() > 0)
                         ? static_cast<int>(pair_i_buf.dimensions()[0]) : 0;
    const int Ms = (world_spheres.dimensions().size() > 0)
                   ? static_cast<int>(world_spheres.dimensions()[0]) : 0;
    const int Mc = (world_capsules.dimensions().size() > 0)
                   ? static_cast<int>(world_capsules.dimensions()[0]) : 0;
    const int Mb = (world_boxes.dimensions().size() > 0)
                   ? static_cast<int>(world_boxes.dimensions()[0]) : 0;
    const int Mh = (world_halfspaces.dimensions().size() > 0)
                   ? static_cast<int>(world_halfspaces.dimensions()[0]) : 0;

    // Per-thread workspace layout (in floats):
    //   10 * T*n_act  (traj, q_k, g, g_prev, dir, best_x, cur_g, x_trial, g_trial, g_dummy)
    //   T * SCO_MAX_G                        (d_k)
    //   T * SCO_MAX_G * n_act                (J_k)
    //   2 * m_lbfgs * T * n_act              (s_buf + y_buf)
    //   2 * SCO_MAX_M                        (rho_buf + alpha_b)
    //   SCO_MAX_N * 7                        (T_world)
    //   n_act                                (q_tmp)
    const int n = T * n_act;
    const int workspace_stride =
        10 * n
        + T * SCO_MAX_G
        + T * SCO_MAX_G * n_act
        + 2 * static_cast<int>(m_lbfgs) * n
        + 2 * SCO_MAX_M
        + SCO_MAX_N * 7
        + n_act;

    constexpr int THREADS = 32;
    const int blocks = (B + THREADS - 1) / THREADS;

    sco_trajopt_kernel<<<blocks, THREADS, 0, stream>>>(
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
        B, T, n_joints, n_act,
        N, static_cast<int>(S), P, Ms, Mc, Mb, Mh,
        static_cast<int>(n_outer_iters),
        static_cast<int>(n_inner_iters),
        static_cast<int>(m_lbfgs),
        w_smooth, w_acc, w_jerk,
        w_limits, w_trust,
        w_collision, w_collision_max, penalty_scale,
        collision_margin, smooth_min_temperature, fd_eps);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // workspace (scratch, discarded)
);
