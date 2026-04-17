/**
 * Sequential Quadratic Programming IK CUDA kernel with XLA FFI binding.
 *
 * Implements multi-seed SQP-IK directly (no coarse phase):
 *   - One CUDA thread per seed.
 *   - Fixed pos_weight / ori_weight.
 *   - Jacobi column scaling in the QP matrices.
 *   - Box-constrained QP solved by Cholesky (unconstrained Newton step)
 *     then clamped to joint limits, followed by n_inner_iters active-set
 *     refinement steps that fix bound-hitting joints and re-solve.
 *   - 5-point line search (early exit on sufficient descent).
 *   - Trust-region step-size schedule.
 *   - All-time best-config tracking.
 *   - Multi-EE support: stacked residuals and Jacobians for all EEs.
 *
 * Algorithmic difference from LS-IK
 *   The LM unconstrained Cholesky solve is replaced by an inner projected
 *   gradient loop that enforces joint limits as hard constraints on the step.
 *   This means the step p always satisfies lower <= q+p <= upper, rather
 *   than clamping q+p after the fact.
 *
 * Reuses _ik_cuda_helpers.cuh for SE(3) math, FK, and IK helpers
 * (residual/Jacobian, Cholesky, small math).
 *
 * Numerical stability:
 *   - FK and Jacobian in float32.
 *   - Normal-equation matrix H and gradient g in float64.
 *   - Inner projected gradient loop in float32.
 *
 * Build with:
 *   bash src/pyroffi/cuda_kernels/build_sqp_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "_collision_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// SQP-IK kernel — one thread per seed
// ---------------------------------------------------------------------------

/**
 * Multi-seed SQP-IK with multi-EE support.
 *
 * Each thread independently refines one seed for max_iter outer iterations.
 * Each outer iteration solves a box-constrained QP via n_inner_iters steps
 * of projected gradient descent, then applies a line search.
 *
 * @param seeds         (n_problems, n_seeds, n_act)   initial configurations
 * @param target_jnts   (n_ee,)                        joint index per EE
 * @param ancestor_masks (n_ee, n_joints)              ancestor bitmask per EE
 * @param target_Ts     (n_problems, n_ee, 7)          target poses
 * @param lower/upper   (n_act,)                       joint limits
 * @param fixed_mask    (n_act,) int32                 1 = frozen joint
 * @param out           (n_problems, n_seeds, n_act)   best configurations
 * @param out_err       (n_problems, n_seeds)          best weighted sq. errors
 * @param n_inner_iters int                            projected-gradient steps
 * @param pos_weight    scalar                         weight on position residual
 * @param ori_weight    scalar                         weight on orientation residual
 * @param lambda_init   scalar                         initial damping factor
 * @param eps_pos       scalar                         position convergence [m]
 * @param eps_ori       scalar                         orientation convergence [rad]
 * @param max_iter      int                            outer SQP iteration budget
 */
__global__
void sqp_ik_kernel(
    const float* __restrict__ seeds,
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
    const float* __restrict__ robot_spheres_local,
    const int*   __restrict__ robot_sphere_joint_idx,
    const float* __restrict__ world_spheres,
    const float* __restrict__ world_capsules,
    const float* __restrict__ world_boxes,
    const float* __restrict__ world_halfspaces,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    int   n_problems, int n_seeds, int n_joints, int n_act, int n_ee,
    int   n_robot_spheres, int n_world_spheres, int n_world_capsules,
    int   n_world_boxes, int n_world_halfspaces,
    int   max_iter, int n_inner_iters,
    int   enable_collision,
    float collision_weight, float collision_margin,
    float pos_weight, float ori_weight, float lambda_init,
    float eps_pos, float eps_ori)
{
    // ── Shared memory: robot parameters loaded once per block ───────────────
    __shared__ float s_twists        [MAX_JOINTS * 6];
    __shared__ float s_parent_tf     [MAX_JOINTS * 7];
    __shared__ int   s_parent_idx    [MAX_JOINTS];
    __shared__ int   s_act_idx       [MAX_JOINTS];
    __shared__ float s_mimic_mul     [MAX_JOINTS];
    __shared__ float s_mimic_off     [MAX_JOINTS];
    __shared__ int   s_mimic_act_idx [MAX_JOINTS];
    __shared__ int   s_topo_inv      [MAX_JOINTS];
    __shared__ float s_target_Ts     [MAX_EE * 7];
    __shared__ int   s_target_jnts   [MAX_EE];
    __shared__ int   s_ancestor_masks[MAX_EE * MAX_JOINTS];
    __shared__ float s_lower   [MAX_ACT];
    __shared__ float s_upper   [MAX_ACT];
    __shared__ int   s_fixed_mask[MAX_ACT];

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
    __syncthreads();

    const int s  = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int gs = p * n_seeds + s;

    // ── Thread-private weight vector ─────────────────────────────────────
    float W[6];
    W[0] = pos_weight; W[1] = pos_weight; W[2] = pos_weight;
    W[3] = ori_weight; W[4] = ori_weight; W[5] = ori_weight;

    // ── Thread-private state ─────────────────────────────────────────────
    float cfg[MAX_ACT], best_cfg[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    for (int a = 0; a < n_act; a++) cfg[a] = seeds[gs * n_act + a];
    for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];

    // Initial weighted error.
    compute_multi_ee_residual_and_jacobian(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_jnts, s_ancestor_masks, s_target_Ts,
        n_joints, n_act, n_ee, r, J);
    float best_err = 0.0f;
    for (int ee = 0; ee < n_ee; ee++)
        for (int k = 0; k < 6; k++) { float rw = r[ee*6+k] * W[k]; best_err += rw * rw; }

    auto collision_penalty = [&](const float* cfg_eval, float* T_eval) {
        if (!enable_collision || n_robot_spheres <= 0) return 0.0f;

        fk_single(
            cfg_eval,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            T_eval,
            n_joints, n_act);

        float pen = 0.0f;
        for (int i = 0; i < n_robot_spheres; i++) {
            const int jidx = robot_sphere_joint_idx[i];
            if (jidx < 0 || jidx >= n_joints) continue;

            const float* sp = robot_spheres_local + i * 4;
            float local_p[3] = {sp[0], sp[1], sp[2]};
            float world_p[3];
            apply_se3_point(T_eval + jidx * 7, local_p, world_p);
            const float rr = sp[3];

            for (int m = 0; m < n_world_spheres; m++) {
                const float* o = world_spheres + m * 4;
                const float d = sphere_sphere_dist(world_p[0], world_p[1], world_p[2], rr,
                                                   o[0], o[1], o[2], o[3]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_capsules; m++) {
                const float* o = world_capsules + m * 7;
                const float d = sphere_capsule_dist(world_p[0], world_p[1], world_p[2], rr,
                                                    o[0], o[1], o[2], o[3], o[4], o[5], o[6]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_boxes; m++) {
                const float* o = world_boxes + m * 15;
                const float d = sphere_box_dist(world_p[0], world_p[1], world_p[2], rr,
                                                o[0], o[1], o[2],
                                                o[3], o[4], o[5],
                                                o[6], o[7], o[8],
                                                o[9], o[10], o[11],
                                                o[12], o[13], o[14]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_halfspaces; m++) {
                const float* o = world_halfspaces + m * 6;
                const float d = sphere_halfspace_dist(world_p[0], world_p[1], world_p[2], rr,
                                                      o[0], o[1], o[2], o[3], o[4], o[5]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
        }
        return collision_weight * pen;
    };

    best_err += collision_penalty(cfg, T_world);

    float lam = lambda_init;

    for (int iter = 0; iter < max_iter; iter++) {

        // ── Residual + Jacobian ─────────────────────────────────────────
        compute_multi_ee_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_target_jnts, s_ancestor_masks, s_target_Ts,
            n_joints, n_act, n_ee, r, J);

        // Early exit if ALL EEs converged.
        {
            bool all_conv = true;
            for (int ee = 0; ee < n_ee; ee++) {
                if (norm3(r + ee*6) >= eps_pos || norm3(r + ee*6 + 3) >= eps_ori) {
                    all_conv = false;
                    break;
                }
            }
            if (all_conv) break;
        }

        // Apply weights to residual and Jacobian rows.
        float fw[6 * MAX_EE];
        for (int k = 0; k < 6 * n_ee; k++) fw[k] = r[k] * W[k % 6];
        for (int ee = 0; ee < n_ee; ee++)
            for (int k = 0; k < 6; k++)
                for (int a = 0; a < n_act; a++)
                    J[(ee*6+k)*n_act+a] *= W[k];

        float curr_err = 0.0f;
        for (int k = 0; k < 6 * n_ee; k++) curr_err += fw[k] * fw[k];
        curr_err += collision_penalty(cfg, T_world);

        // ── Jacobi column scaling ───────────────────────────────────────
        float col_scale[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float sq = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) { float v = J[k*n_act+a]; sq += v*v; }
            col_scale[a] = sqrtf(sq) + 1e-8f;
        }
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k*n_act+a] /= col_scale[a];

        // ── Form H_s = Js^T Js + λI and g_s = Js^T fw (float64) ───────
        double H_s[MAX_ACT * MAX_ACT];
        double g_s[MAX_ACT];

        for (int i = 0; i < n_act; i++) {
            for (int j = 0; j < n_act; j++) {
                double acc = 0.0;
                for (int k = 0; k < 6 * n_ee; k++)
                    acc += (double)J[k*n_act+i] * (double)J[k*n_act+j];
                H_s[i*n_act + j] = acc;
            }
            double gb = 0.0;
            for (int k = 0; k < 6 * n_ee; k++)
                gb += (double)J[k*n_act+i] * (double)fw[k];
            g_s[i] = gb;
            H_s[i*n_act + i] += (double)lam;
        }

        // ── Box bounds in scaled space ──────────────────────────────────
        float lb_s[MAX_ACT], ub_s[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            lb_s[a] = (s_lower[a] - cfg[a]) * col_scale[a];
            ub_s[a] = (s_upper[a] - cfg[a]) * col_scale[a];
            if (s_fixed_mask[a]) { lb_s[a] = 0.0f; ub_s[a] = 0.0f; }
        }

        // ── Step 1: unconstrained Newton step via Cholesky (same as LM) ─
        // chol_solve modifies A/b in-place; preserve H_s and g_s for the
        // active-set refinement steps that follow.
        double A_init[MAX_ACT * MAX_ACT];
        double rhs_init[MAX_ACT];
        for (int i = 0; i < n_act * n_act; i++) A_init[i] = H_s[i];
        for (int a = 0; a < n_act; a++) rhs_init[a] = -g_s[a];

        // Handle fixed joints: unit row/col, zero rhs.
        for (int a = 0; a < n_act; a++) {
            if (!s_fixed_mask[a]) continue;
            for (int j = 0; j < n_act; j++) A_init[a*n_act+j] = A_init[j*n_act+a] = 0.0;
            A_init[a*n_act+a] = 1.0;
            rhs_init[a] = 0.0;
        }
        chol_solve(A_init, rhs_init, n_act);

        float p_s[MAX_ACT];
        for (int a = 0; a < n_act; a++) p_s[a] = (float)rhs_init[a];

        // ── Step 2: project to joint-limit box ──────────────────────────
        for (int a = 0; a < n_act; a++)
            p_s[a] = clampf(p_s[a], lb_s[a], ub_s[a]);

        // ── Active-set refinement steps ─────────────────────────────────
        // Fix joints that hit their bounds, re-solve for free joints.
        // Converges in 1-2 steps; no-op when joints are not near limits.
        for (int k = 0; k < n_inner_iters; k++) {
            float active[MAX_ACT], p_bounded[MAX_ACT];
            for (int a = 0; a < n_act; a++) {
                active[a]    = (p_s[a] <= lb_s[a] + 1e-8f || p_s[a] >= ub_s[a] - 1e-8f)
                               ? 1.0f : 0.0f;
                p_bounded[a] = clampf(p_s[a], lb_s[a], ub_s[a]) * active[a];
            }

            // g_adj = g_s + H_s @ p_bounded; masked system for free joints.
            double A_ref[MAX_ACT * MAX_ACT];
            double rhs_ref[MAX_ACT];
            for (int a = 0; a < n_act; a++) {
                double adj = 0.0;
                for (int b = 0; b < n_act; b++)
                    adj += H_s[a*n_act+b] * (double)p_bounded[b];
                rhs_ref[a] = -(g_s[a] + adj) * (double)(1.0f - active[a]);

                for (int b = 0; b < n_act; b++) {
                    A_ref[a*n_act+b] = (active[a] > 0.5f || active[b] > 0.5f)
                                       ? 0.0 : H_s[a*n_act+b];
                }
                if (active[a] > 0.5f) A_ref[a*n_act+a] = 1.0;
            }
            chol_solve(A_ref, rhs_ref, n_act);

            for (int a = 0; a < n_act; a++) {
                p_s[a] = (active[a] > 0.5f)
                         ? p_bounded[a]
                         : clampf((float)rhs_ref[a], lb_s[a], ub_s[a]);
            }
        }

        // ── Unscale to original joint space ─────────────────────────────
        float delta[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            delta[a] = p_s[a] / col_scale[a];

        // ── Trust-region step clipping ──────────────────────────────────
        {
            float max_p = 0.0f, max_o = 0.0f;
            for (int ee = 0; ee < n_ee; ee++) {
                max_p = fmaxf(max_p, norm3(r + ee*6));
                max_o = fmaxf(max_o, norm3(r + ee*6 + 3));
            }
            float R;
            if      (max_p > 1e-2f || max_o > 0.6f)  R = 0.38f;
            else if (max_p > 1e-3f || max_o > 0.25f) R = 0.22f;
            else if (max_p > 2e-4f || max_o > 0.08f) R = 0.12f;
            else                                       R = 0.05f;

            float dnorm = 0.0f;
            for (int a = 0; a < n_act; a++) dnorm += delta[a]*delta[a];
            dnorm = sqrtf(dnorm);
            if (dnorm > R) {
                const float scale = R / (dnorm + 1e-18f);
                for (int a = 0; a < n_act; a++) delta[a] *= scale;
            }
        }

        // ── Line search over 5 step sizes ──────────────────────────────
        const float alphas[5] = { 1.0f, 0.5f, 0.25f, 0.1f, 0.025f };
        float best_alpha_err = 1e30f;
        int   best_alpha_idx = 0;
        float r_trial[6 * MAX_EE];

        for (int ai = 0; ai < 5; ai++) {
            float cfg_trial[MAX_ACT];
            for (int a = 0; a < n_act; a++)
                cfg_trial[a] = clampf(cfg[a] + alphas[ai] * delta[a],
                                      s_lower[a], s_upper[a]);

            compute_multi_ee_residual_only(
                cfg_trial, T_world,
                s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r_trial);

            float err_trial = 0.0f;
            for (int ee = 0; ee < n_ee; ee++)
                for (int k = 0; k < 6; k++) {
                    float rw = r_trial[ee*6+k] * W[k];
                    err_trial += rw * rw;
                }
            err_trial += collision_penalty(cfg_trial, T_world);
            if (err_trial < best_alpha_err) {
                best_alpha_err = err_trial;
                best_alpha_idx = ai;
            }
        }

        // Best trial configuration.
        float trial_cfg[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            trial_cfg[a] = clampf(cfg[a] + alphas[best_alpha_idx] * delta[a],
                                  s_lower[a], s_upper[a]);

        // ── Accept / reject ─────────────────────────────────────────────
        const bool improved = best_alpha_err < curr_err * (1.0f - 1e-4f);
        if (improved) {
            for (int a = 0; a < n_act; a++) cfg[a] = trial_cfg[a];
            lam = fmaxf(lam * 0.5f, 1e-10f);
        } else {
            lam = fminf(lam * 3.0f, 1e6f);
        }

        // ── Track all-time best ─────────────────────────────────────────
        if (best_alpha_err < best_err) {
            best_err = best_alpha_err;
            for (int a = 0; a < n_act; a++) best_cfg[a] = trial_cfg[a];
        }
    }

    // Write output.
    for (int a = 0; a < n_act; a++) out[gs * n_act + a] = best_cfg[a];
    out_err[gs] = best_err;
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error SqpIkCudaImpl(
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
    ffi::Buffer<ffi::DataType::F32> robot_spheres_local,
    ffi::Buffer<ffi::DataType::S32> robot_sphere_joint_idx,
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t max_iter,
    int64_t n_inner_iters,
    float   pos_weight,
    float   ori_weight,
    float   lambda_init,
    float   eps_pos,
    float   eps_ori,
    int64_t enable_collision,
    float   collision_weight,
    float   collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    const int n_act      = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);
    const int n_ee       = static_cast<int>(target_jnts.dimensions()[0]);
    const int n_robot_spheres = static_cast<int>(robot_spheres_local.dimensions()[0]);
    const int n_world_spheres = static_cast<int>(world_spheres.dimensions()[0]);
    const int n_world_capsules = static_cast<int>(world_capsules.dimensions()[0]);
    const int n_world_boxes = static_cast<int>(world_boxes.dimensions()[0]);
    const int n_world_halfspaces = static_cast<int>(world_halfspaces.dimensions()[0]);

    // SQP is more register-heavy than LS due to the inner H_s/g_s buffers.
    constexpr int THREADS_MAX = 32;
    const int threads  = n_seeds < THREADS_MAX ? n_seeds : THREADS_MAX;
    const int blocks_x = (n_seeds + threads - 1) / threads;

    sqp_ik_kernel<<<dim3(blocks_x, n_problems), threads, 0, stream>>>(
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
        robot_spheres_local.typed_data(),
        robot_sphere_joint_idx.typed_data(),
        world_spheres.typed_data(),
        world_capsules.typed_data(),
        world_boxes.typed_data(),
        world_halfspaces.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        out->typed_data(),
        out_err->typed_data(),
        n_problems, n_seeds, n_joints, n_act, n_ee,
        n_robot_spheres, n_world_spheres, n_world_capsules,
        n_world_boxes, n_world_halfspaces,
        static_cast<int>(max_iter),
        static_cast<int>(n_inner_iters),
        static_cast<int>(enable_collision),
        collision_weight,
        collision_margin,
        pos_weight, ori_weight, lambda_init,
        eps_pos, eps_ori);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SqpIkCudaFfi, SqpIkCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // target_jnts
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // ancestor_masks
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // target_Ts
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // robot_spheres_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // robot_sphere_joint_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // fixed_mask
        .Attr<int64_t>("max_iter")
        .Attr<int64_t>("n_inner_iters")
        .Attr<float>("pos_weight")
        .Attr<float>("ori_weight")
        .Attr<float>("lambda_init")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out cfgs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out errors
);
