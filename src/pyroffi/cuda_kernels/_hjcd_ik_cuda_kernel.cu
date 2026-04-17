/**
 * HJCD-IK CUDA kernel with XLA FFI binding.
 *
 * Implements the two-phase HJCD-IK algorithm in CUDA:
 *
 *   Phase 1 (Coarse):  Greedy coordinate-descent — selects the single
 *                      joint with maximum predicted error reduction per step.
 *   Phase 2 (Refine):  Levenberg-Marquardt — column-scaled normal equations
 *                      with joint-limit prior, line search, stall kicks.
 *
 * Multi-EE support: stacked residuals and Jacobians for all EEs simultaneously.
 * FK is called once per LM iteration; each EE reads from the FK result.
 *
 * FK and shared IK helpers are provided via _ik_cuda_helpers.cuh so that
 * FK/residual/Jacobian logic is not duplicated.
 *
 * Numerical stability:
 *   - FK and Jacobian in float32.
 *   - Normal-equation matrix and Cholesky solve in float64.
 *   - All kernel launches are associated with the caller's CUDA stream so
 *     there are no implicit device synchronisations.
 *
 * Build with:  bash src/pyroffi/cuda_kernels/build_hjcd_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "_collision_cuda_helpers.cuh"

#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits
// ---------------------------------------------------------------------------

// Defined in _ik_cuda_helpers.cuh (override by defining before include).

// ---------------------------------------------------------------------------
// Math helpers (IK-specific)
// ---------------------------------------------------------------------------

/** Dot product of two 3-vectors. */
__device__ __forceinline__
float dot3(const float* __restrict__ a, const float* __restrict__ b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// ---------------------------------------------------------------------------
// Adaptive weighting (matches JAX _adaptive_weights)
// ---------------------------------------------------------------------------

/**
 * Compute per-residual adaptive weights.
 * w[0:3] = 1, w[3:6] = clamp(pos_err / ori_err, 0.05, 1.0).
 */
__device__ void adaptive_weights(const float* __restrict__ r, float* __restrict__ w)
{
    const float pos_err = norm3(r)     + 1e-8f;
    const float ori_err = norm3(r + 3) + 1e-8f;
    const float ori_scale = clampf(pos_err / ori_err, 0.05f, 1.0f);
    w[0] = 1.0f; w[1] = 1.0f; w[2] = 1.0f;
    w[3] = ori_scale; w[4] = ori_scale; w[5] = ori_scale;
}

// ---------------------------------------------------------------------------
// Coarse greedy coordinate-descent kernel (Phase 1)
// ---------------------------------------------------------------------------

/**
 * One CUDA thread per seed.  Runs k_max greedy CD steps.
 *
 * Multi-EE: stacked residuals and Jacobians; per-EE adaptive weights;
 * CD selects best joint based on all EEs combined.
 *
 * @param seeds          (n_problems, n_seeds, n_act)
 * @param target_jnts    (n_ee,)                       joint index per EE
 * @param ancestor_masks (n_ee, n_joints)               ancestor bitmask per EE
 * @param target_Ts      (n_problems, n_ee, 7)          target poses
 * @param lower          (n_act,) lower limits
 * @param upper          (n_act,) upper limits
 * @param fixed_mask     (n_act,) int32; 1 = fixed, 0 = free
 * @param out            (n_problems, n_seeds, n_act)  output configurations
 * @param n_ee           number of end-effectors
 */
__global__
void hjcd_ik_coarse_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ target_jnts,     // (n_ee,) NEW
    const int*   __restrict__ ancestor_masks,  // (n_ee, n_joints) NEW
    const float* __restrict__ target_Ts,       // (n_problems, n_ee, 7) NEW
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
    int n_problems, int n_seeds, int n_joints, int n_act, int n_ee,
    int n_robot_spheres, int n_world_spheres, int n_world_capsules,
    int n_world_boxes, int n_world_halfspaces,
    int k_max, int enable_collision, float collision_weight, float collision_margin)
{
    // ── Shared memory: robot parameters loaded once per block ───────────────
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

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int gs = p * n_seeds + s;

    // Local configuration (thread-private).
    float cfg[MAX_ACT];
    for (int a = 0; a < n_act; a++) cfg[a] = seeds[gs * n_act + a];

    // Scratch for FK world transforms and stacked Jacobian.
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    for (int iter = 0; iter < k_max; iter++) {
        compute_multi_ee_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_target_jnts, s_ancestor_masks, s_target_Ts,
            n_joints, n_act, n_ee, r, J);

        // Per-EE adaptive weights with orientation gating (pos < 1mm).
        float w[6 * MAX_EE];
        for (int ee = 0; ee < n_ee; ee++) {
            float w_ee[6];
            adaptive_weights(r + ee*6, w_ee);
            const float pe = norm3(r + ee*6);
            if (pe >= 1e-3f) { w_ee[3] = 0.0f; w_ee[4] = 0.0f; w_ee[5] = 0.0f; }
            for (int k = 0; k < 6; k++) w[ee*6+k] = w_ee[k];
        }

        // fw = r * w (stacked), apply weights in-place to J rows.
        float fw[6 * MAX_EE];
        for (int k = 0; k < 6 * n_ee; k++) fw[k] = r[k] * w[k];
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k * n_act + a] *= w[k];

        // Per-joint: JwTfw[a] = Jw[:,a]^T fw,  Jw_normsq[a] = ||Jw[:,a]||^2.
        float JwTfw[MAX_ACT], Jw_normsq[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float s_dot = 0.0f, s_sq = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) {
                const float jwa = J[k * n_act + a];
                s_dot += jwa * fw[k];
                s_sq  += jwa * jwa;
            }
            JwTfw[a]     = s_dot;
            Jw_normsq[a] = s_sq + 1e-8f;
        }

        // Find best joint.
        int best = -1;
        float best_impr = -1.0f;
        for (int a = 0; a < n_act; a++) {
            if (s_fixed_mask[a]) continue;
            const float impr = JwTfw[a] * JwTfw[a] / Jw_normsq[a];
            if (impr > best_impr) { best_impr = impr; best = a; }
        }
        if (best < 0) break;

        // Apply step for best joint only.
        const float step = -JwTfw[best] / Jw_normsq[best];
        cfg[best] = clampf(cfg[best] + step, s_lower[best], s_upper[best]);
    }

    // Compute final unweighted error for scoring (sum over all EEs).
    compute_multi_ee_residual_only(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r);
    float final_err = 0.0f;
    for (int k = 0; k < 6 * n_ee; k++) final_err += r[k] * r[k];

    if (enable_collision && n_robot_spheres > 0) {
        fk_single(
            cfg,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            T_world,
            n_joints, n_act);

        float pen = 0.0f;
        for (int i = 0; i < n_robot_spheres; i++) {
            const int jidx = robot_sphere_joint_idx[i];
            if (jidx < 0 || jidx >= n_joints) continue;

            const float* sp = robot_spheres_local + i * 4;
            float local_p[3] = {sp[0], sp[1], sp[2]};
            float world_p[3];
            apply_se3_point(T_world + jidx * 7, local_p, world_p);
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
        final_err += collision_weight * pen;
    }
    out_err[gs] = final_err;

    // Write output.
    for (int a = 0; a < n_act; a++) out[gs * n_act + a] = cfg[a];
}

// ---------------------------------------------------------------------------
// LM refinement kernel (Phase 2)
// ---------------------------------------------------------------------------

/**
 * One CUDA thread per refinement seed.  Runs max_iter LM iterations with:
 *   - Per-EE adaptive pos/ori row weighting.
 *   - Jacobi column scaling.
 *   - Soft joint-limit prior.
 *   - Line search over {1, 0.5, 0.25, 0.1, 0.025} step multipliers.
 *   - Stall detection with random kicks.
 *   - Best-config tracking.
 *   - Multi-EE: convergence requires ALL EEs satisfied.
 *
 * @param seeds         (n_problems, n_seeds, n_act)
 * @param noise         (n_problems, n_seeds, max_iter, n_act)  kick noise
 * @param target_jnts   (n_ee,)                       joint index per EE
 * @param ancestor_masks (n_ee, n_joints)              ancestor bitmask per EE
 * @param target_Ts     (n_problems, n_ee, 7)          target poses
 */
__global__
void hjcd_ik_lm_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ noise,         // (n_problems, n_seeds, max_iter, n_act)
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ target_jnts,     // (n_ee,) NEW
    const int*   __restrict__ ancestor_masks,  // (n_ee, n_joints) NEW
    const float* __restrict__ target_Ts,       // (n_problems, n_ee, 7) NEW
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
    int*         __restrict__ stop_flag,
    int n_problems, int n_seeds, int n_joints, int n_act, int n_ee, int max_iter,
    int n_robot_spheres, int n_world_spheres, int n_world_capsules,
    int n_world_boxes, int n_world_halfspaces,
    float lambda_init, float limit_prior_weight, float kick_scale,
    float eps_pos, float eps_ori, int stall_patience,
    int enable_collision, float collision_weight, float collision_margin)
{
    // ── Shared memory: robot parameters loaded once per block ───────────────
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

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int gs = p * n_seeds + s;

    // Thread-private state.
    float cfg[MAX_ACT], best_cfg[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];

    // Load initial config.
    for (int a = 0; a < n_act; a++) cfg[a] = seeds[gs * n_act + a];
    for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];

    // Joint-limit mid / half-range for prior.
    float mid[MAX_ACT], half_range[MAX_ACT];
    for (int a = 0; a < n_act; a++) {
        mid[a]        = (s_lower[a] + s_upper[a]) * 0.5f;
        half_range[a] = (s_upper[a] - s_lower[a]) * 0.5f + 1e-8f;
    }

    // Compute initial unweighted squared error (sum over all EEs).
    compute_multi_ee_residual_and_jacobian(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_jnts, s_ancestor_masks, s_target_Ts,
        n_joints, n_act, n_ee, r, J);
    float best_err = 0.0f;
    for (int k = 0; k < 6 * n_ee; k++) best_err += r[k] * r[k];

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

    float lam   = lambda_init;
    int   stall = 0;
    bool  done  = false;

    for (int iter = 0; iter < max_iter; iter++) {
        if (done) break;
        if (stop_flag[p]) break;  // Another seed in this problem converged

        // ── Jacobian + residual ──────────────────────────────────────────
        compute_multi_ee_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_target_jnts, s_ancestor_masks, s_target_Ts,
            n_joints, n_act, n_ee, r, J);

        // Unweighted current error (sum over all EEs).
        float curr_err = 0.0f;
        for (int k = 0; k < 6 * n_ee; k++) curr_err += r[k] * r[k];
        curr_err += collision_penalty(cfg, T_world);

        // Early exit check: ALL EEs must converge.
        {
            bool all_conv = true;
            for (int ee = 0; ee < n_ee; ee++) {
                float r_pos = norm3(r + ee*6);
                float r_ori = norm3(r + ee*6 + 3);
                if (r_pos >= eps_pos || r_ori >= eps_ori) { all_conv = false; break; }
            }
            if (all_conv) {
                done = true;
                atomicExch(stop_flag + p, 1);
                __threadfence();
                break;
            }
        }

        // ── Per-EE adaptive row weighting ────────────────────────────────
        float w[6 * MAX_EE];
        for (int ee = 0; ee < n_ee; ee++) {
            float w_ee[6];
            adaptive_weights(r + ee*6, w_ee);
            for (int k = 0; k < 6; k++) w[ee*6+k] = w_ee[k];
        }
        float fw[6 * MAX_EE];
        for (int k = 0; k < 6 * n_ee; k++) fw[k] = r[k] * w[k];
        // Apply weights in-place to J rows.
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k * n_act + a] *= w[k];

        // ── Jacobi column scaling ─────────────────────────────────────────
        float col_scale[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float sq = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) { float v = J[k*n_act+a]; sq += v*v; }
            col_scale[a] = sqrtf(sq) + 1e-8f;
        }
        // Scale J in-place → Js (reuse J buffer).
        for (int k = 0; k < 6 * n_ee; k++)
            for (int a = 0; a < n_act; a++)
                J[k * n_act + a] /= col_scale[a];

        // ── Normal equations (float64 for numerical stability) ───────────
        double A_s[MAX_ACT * MAX_ACT];
        double rhs_s[MAX_ACT];

        for (int i = 0; i < n_act; i++) {
            for (int j = 0; j < n_act; j++) {
                double acc = 0.0;
                for (int k = 0; k < 6 * n_ee; k++)
                    acc += (double)J[k*n_act+i] * (double)J[k*n_act+j];
                A_s[i*n_act + j] = acc;
            }
            double rb = 0.0;
            for (int k = 0; k < 6 * n_ee; k++)
                rb += (double)J[k*n_act+i] * (double)fw[k];
            rhs_s[i] = rb;
        }

        // Add joint-limit prior and LM damping (in scaled space).
        for (int a = 0; a < n_act; a++) {
            const double D_prior_raw  = (double)limit_prior_weight /
                                        ((double)half_range[a] * (double)half_range[a]);
            const double cs2          = (double)col_scale[a] * (double)col_scale[a];
            const double D_prior_s    = D_prior_raw / cs2;
            const double g_prior_s    = D_prior_raw * (double)(cfg[a] - mid[a])
                                        / (double)col_scale[a];
            A_s[a*n_act + a] += (double)lam + D_prior_s;
            rhs_s[a]          = -(rhs_s[a] + g_prior_s);
        }

        // Mask fixed joints: zero row and col, set diagonal to 1, rhs to 0.
        for (int a = 0; a < n_act; a++) {
            if (!s_fixed_mask[a]) continue;
            for (int j = 0; j < n_act; j++) A_s[a*n_act+j] = A_s[j*n_act+a] = 0.0;
            A_s[a*n_act+a] = 1.0;
            rhs_s[a] = 0.0;
        }

        // Solve (overwrites A_s and rhs_s; solution in rhs_s).
        chol_solve(A_s, rhs_s, n_act);

        // Unscale: delta_a = p_s[a] / col_scale[a].
        float delta[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            delta[a] = (float)rhs_s[a] / col_scale[a];

        // Trust-region step clipping: use MAX error across all EEs.
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

        // ── Line search: 5 candidates with unweighted error ───────────────
        const float alphas[5] = { 1.0f, 0.5f, 0.25f, 0.1f, 0.025f };
        float best_alpha_err  = 1e30f;
        int   best_alpha_idx  = 0;
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
            for (int k = 0; k < 6 * n_ee; k++) err_trial += r_trial[k] * r_trial[k];
            err_trial += collision_penalty(cfg_trial, T_world);
            if (err_trial < best_alpha_err) {
                best_alpha_err = err_trial;
                best_alpha_idx = ai;
                if (err_trial < curr_err * (1.0f - 1e-4f)) break;
            }
        }

        float trial_cfg[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            trial_cfg[a] = clampf(cfg[a] + alphas[best_alpha_idx] * delta[a],
                                  s_lower[a], s_upper[a]);

        // Accept / reject.
        const bool improved = best_alpha_err < curr_err * (1.0f - 1e-4f);
        if (improved) {
            for (int a = 0; a < n_act; a++) cfg[a] = trial_cfg[a];
            lam = fmaxf(lam * 0.5f, 1e-10f);
            stall = 0;
        } else {
            lam = fminf(lam * 3.0f, 1e6f);
            stall++;
        }

        // ── Track all-time best ─────────────────────────────────────────
        if (best_alpha_err < best_err) {
            best_err = best_alpha_err;
            for (int a = 0; a < n_act; a++) best_cfg[a] = trial_cfg[a];
        }

        // ── Stall kick ──────────────────────────────────────────────────
        if (stall >= stall_patience) {
            const float* kick_noise = noise + ((p * n_seeds + s) * max_iter + iter) * n_act;
            for (int a = 0; a < n_act; a++) {
                if (s_fixed_mask[a]) continue;
                cfg[a] = clampf(cfg[a] + kick_noise[a] * kick_scale,
                                s_lower[a], s_upper[a]);
            }
            lam   = lambda_init;
            stall = 0;
        }
    }

    // Write best-seen config and error.
    for (int a = 0; a < n_act; a++) out[gs * n_act + a] = best_cfg[a];
    out_err[gs] = best_err;
}

// ---------------------------------------------------------------------------
// XLA FFI handlers
// ---------------------------------------------------------------------------

static ffi::Error HjcdIkCoarseCudaImpl(
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
    ffi::Buffer<ffi::DataType::S32> target_jnts,     // (n_ee,) NEW
    ffi::Buffer<ffi::DataType::S32> ancestor_masks,  // (n_ee, n_joints) NEW
    ffi::Buffer<ffi::DataType::F32> target_Ts,       // (n_problems, n_ee, 7) NEW
    ffi::Buffer<ffi::DataType::F32> robot_spheres_local,
    ffi::Buffer<ffi::DataType::S32> robot_sphere_joint_idx,
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t k_max,
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

    constexpr int THREADS_MAX = 128;
    const int threads  = n_seeds < THREADS_MAX ? n_seeds : THREADS_MAX;
    const int blocks_x = (n_seeds + threads - 1) / threads;

    hjcd_ik_coarse_kernel<<<dim3(blocks_x, n_problems), threads, 0, stream>>>(
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
        static_cast<int>(k_max),
        static_cast<int>(enable_collision),
        collision_weight,
        collision_margin);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

static ffi::Error HjcdIkLmCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds,
    ffi::Buffer<ffi::DataType::F32> noise,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::S32> target_jnts,     // (n_ee,) NEW
    ffi::Buffer<ffi::DataType::S32> ancestor_masks,  // (n_ee, n_joints) NEW
    ffi::Buffer<ffi::DataType::F32> target_Ts,       // (n_problems, n_ee, 7) NEW
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
    int64_t stall_patience,
    float   lambda_init,
    float   limit_prior_weight,
    float   kick_scale,
    float   eps_pos,
    float   eps_ori,
    int64_t enable_collision,
    float   collision_weight,
    float   collision_margin,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> stop_flag)
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

    constexpr int THREADS_MAX = 32;
    const int threads  = n_seeds < THREADS_MAX ? n_seeds : THREADS_MAX;
    const int blocks_x = (n_seeds + threads - 1) / threads;

    // Zero per-problem stop flags before kernel launch.
    cudaMemsetAsync(stop_flag->typed_data(), 0, n_problems * sizeof(int), stream);

    hjcd_ik_lm_kernel<<<dim3(blocks_x, n_problems), threads, 0, stream>>>(
        seeds.typed_data(),
        noise.typed_data(),
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
        stop_flag->typed_data(),
        n_problems, n_seeds, n_joints, n_act, n_ee,
        static_cast<int>(max_iter),
        n_robot_spheres, n_world_spheres, n_world_capsules,
        n_world_boxes, n_world_halfspaces,
        lambda_init, limit_prior_weight, kick_scale,
        eps_pos, eps_ori,
        static_cast<int>(stall_patience),
        static_cast<int>(enable_collision),
        collision_weight,
        collision_margin);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Handler registration
// ---------------------------------------------------------------------------

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HjcdIkCoarseCudaFfi, HjcdIkCoarseCudaImpl,
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
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // robot_spheres_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // robot_sphere_joint_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fixed_mask
        .Attr<int64_t>("k_max")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out_err

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HjcdIkLmCudaFfi, HjcdIkLmCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // noise
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
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // robot_spheres_local
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // robot_sphere_joint_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fixed_mask
        .Attr<int64_t>("max_iter")
        .Attr<int64_t>("stall_patience")
        .Attr<float>("lambda_init")
        .Attr<float>("limit_prior_weight")
        .Attr<float>("kick_scale")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // out
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // out_err
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // stop_flag
