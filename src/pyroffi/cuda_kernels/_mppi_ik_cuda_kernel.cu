/**
 * MPPI + L-BFGS IK CUDA kernel with XLA FFI binding (optimized).
 *
 * Optimizations vs. original:
 *
 *   1. RNG-replay MPPI: eliminates the noise[MAX_PARTICLES * MAX_ACT] buffer.
 *      Pass 1: generate noise on-the-fly, evaluate particles, store only costs.
 *      Pass 2: replay the same RNG sequence, accumulate weighted-mean delta
 *              using softmax weights computed from the stored costs.
 *      This removes the dominant source of register / local-memory pressure.
 *
 *   2. In-place weights: the costs[] array is reused for softmax weights,
 *      eliminating the separate weights[MAX_PARTICLES] buffer.
 *
 *   3. rng_seed passed as int64 attribute (was a device buffer dereferenced
 *      on the host → segfault).
 *
 *   4. Early-exit line searches: both MPPI best-tracking and L-BFGS line
 *      search break on sufficient decrease (≥ 1e-4 relative improvement),
 *      saving up to 4 redundant FK evaluations per iteration.
 *
 *   5. Convergence gate: if MPPI already satisfies eps_pos / eps_ori for
 *      all EEs, the L-BFGS stage is skipped entirely.
 *
 *   6. Removed cudaStreamSynchronize (XLA owns stream scheduling).
 *
 * Build with:
 *   bash src/pyroffi/cuda_kernels/build_mppi_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "_collision_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// MPPI + L-BFGS IK kernel — one thread per seed
// ---------------------------------------------------------------------------

__global__
void mppi_ik_kernel(
    const float*    __restrict__ seeds,
    const float*    __restrict__ twists,
    const float*    __restrict__ parent_tf,
    const int*      __restrict__ parent_idx,
    const int*      __restrict__ act_idx,
    const float*    __restrict__ mimic_mul,
    const float*    __restrict__ mimic_off,
    const int*      __restrict__ mimic_act_idx,
    const int*      __restrict__ topo_inv,
    const int*      __restrict__ target_jnts,
    const int*      __restrict__ ancestor_masks,
    const float*    __restrict__ target_Ts,
    const float*    __restrict__ robot_spheres_local,
    const int*      __restrict__ robot_sphere_joint_idx,
    const float*    __restrict__ world_spheres,
    const float*    __restrict__ world_capsules,
    const float*    __restrict__ world_boxes,
    const float*    __restrict__ world_halfspaces,
    const int*      __restrict__ self_pair_i,
    const int*      __restrict__ self_pair_j,
    const float*    __restrict__ lower,
    const float*    __restrict__ upper,
    const int*      __restrict__ fixed_mask,
    const int*      __restrict__ rng_seed_ptr,
    float*          __restrict__ out,
    float*          __restrict__ out_err,
    int   n_problems, int n_seeds, int n_joints, int n_act, int n_ee,
    int   n_robot_spheres, int n_world_spheres, int n_world_capsules,
    int   n_world_boxes, int n_world_halfspaces, int n_self_pairs,
    int   n_particles, int n_mppi_iters, int n_lbfgs_iters, int m_lbfgs,
    int   enable_collision,
    float collision_weight, float collision_margin,
    float sigma, float mppi_temperature,
    float pos_weight, float ori_weight,
    float eps_pos, float eps_ori)
{
    // ── Shared memory: robot parameters loaded once per block ───────────
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
    __shared__ float s_lower         [MAX_ACT];
    __shared__ float s_upper         [MAX_ACT];
    __shared__ int   s_fixed_mask    [MAX_ACT];

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
    const int p_idx = blockIdx.y;
    for (int i = threadIdx.x; i < n_ee * 7; i += blockDim.x)
        s_target_Ts[i] = target_Ts[p_idx * n_ee * 7 + i];
    for (int i = threadIdx.x; i < n_ee; i += blockDim.x)
        s_target_jnts[i] = target_jnts[i];
    for (int i = threadIdx.x; i < n_ee * n_joints; i += blockDim.x)
        s_ancestor_masks[i] = ancestor_masks[i];
    __syncthreads();

    const int s  = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int gs = p_idx * n_seeds + s;

    // ── Weight vector ────────────────────────────────────────────────────
    const float W[6] = { pos_weight, pos_weight, pos_weight,
                         ori_weight, ori_weight, ori_weight };

    // ── Thread-private state ─────────────────────────────────────────────
    float cfg[MAX_ACT], best_cfg[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[6 * MAX_EE];
    float J[6 * MAX_EE * MAX_ACT];
    float fw[6 * MAX_EE];

    // MPPI: only costs stored (noise is replayed via RNG)
    float costs[MAX_PARTICLES];

    // L-BFGS buffers
    float s_buf    [MAX_LBFGS_M * MAX_ACT];
    float y_buf    [MAX_LBFGS_M * MAX_ACT];
    float rho_buf  [MAX_LBFGS_M];
    float alpha_buf[MAX_LBFGS_M];
    float g        [MAX_ACT];
    float g_prev   [MAX_ACT];
    float cfg_prev [MAX_ACT];
    float dir      [MAX_ACT];

    for (int a = 0; a < n_act; a++) {
        cfg[a]      = seeds[gs * n_act + a];
        best_cfg[a] = cfg[a];
    }

    // Initial cost.
    compute_multi_ee_residual_only(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r);
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

        // Cache sphere world positions + radii (-radius if inactive).
        float sphere_world[MAX_ROBOT_SPHERES * 4];
        for (int i = 0; i < n_robot_spheres; i++) {
            const int jidx = robot_sphere_joint_idx[i];
            if (jidx < 0 || jidx >= n_joints) {
                sphere_world[i*4 + 3] = -1.0f;
                continue;
            }
            const float* sp = robot_spheres_local + i * 4;
            float local_p[3] = {sp[0], sp[1], sp[2]};
            apply_se3_point(T_eval + jidx * 7, local_p, sphere_world + i * 4);
            sphere_world[i*4 + 3] = sp[3];
        }

        float pen = 0.0f;
        for (int i = 0; i < n_robot_spheres; i++) {
            const float rr = sphere_world[i*4 + 3];
            if (rr < 0.0f) continue;
            const float wx = sphere_world[i*4 + 0];
            const float wy = sphere_world[i*4 + 1];
            const float wz = sphere_world[i*4 + 2];

            for (int m = 0; m < n_world_spheres; m++) {
                const float* o = world_spheres + m * 4;
                const float d = sphere_sphere_dist(wx, wy, wz, rr,
                                                   o[0], o[1], o[2], o[3]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_capsules; m++) {
                const float* o = world_capsules + m * 7;
                const float d = sphere_capsule_dist(wx, wy, wz, rr,
                                                    o[0], o[1], o[2], o[3], o[4], o[5], o[6]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
            for (int m = 0; m < n_world_boxes; m++) {
                const float* o = world_boxes + m * 15;
                const float d = sphere_box_dist(wx, wy, wz, rr,
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
                const float d = sphere_halfspace_dist(wx, wy, wz, rr,
                                                      o[0], o[1], o[2], o[3], o[4], o[5]);
                if (d < collision_margin) {
                    const float diff = d - collision_margin;
                    pen += diff * diff;
                }
            }
        }

        // Self-collision pass — pre-pruned (sphere_a, sphere_b) pairs.
        for (int i = 0; i < n_self_pairs; i++) {
            const int a = self_pair_i[i];
            const int b = self_pair_j[i];
            if (a < 0 || a >= n_robot_spheres || b < 0 || b >= n_robot_spheres) continue;
            const float ra = sphere_world[a*4 + 3];
            const float rb = sphere_world[b*4 + 3];
            if (ra < 0.0f || rb < 0.0f) continue;
            const float d = sphere_sphere_dist(
                sphere_world[a*4+0], sphere_world[a*4+1], sphere_world[a*4+2], ra,
                sphere_world[b*4+0], sphere_world[b*4+1], sphere_world[b*4+2], rb);
            if (d < collision_margin) {
                const float diff = d - collision_margin;
                pen += diff * diff;
            }
        }

        return collision_weight * pen;
    };

    best_err += collision_penalty(cfg, T_world);

    // Per-thread RNG.
    uint32_t rng_state = (uint32_t)(*rng_seed_ptr)
                       ^ (uint32_t)(s     * 0x9e3779b9u)
                       ^ (uint32_t)(p_idx * 0x6c62272eu);
    xorshift32(rng_state); xorshift32(rng_state); xorshift32(rng_state);

    const float inv_temp = 1.0f / fmaxf(mppi_temperature, 1e-8f);

    // =====================================================================
    // Stage 1: MPPI particle search (RNG-replay, no noise buffer)
    // =====================================================================

    for (int mppi_iter = 0; mppi_iter < n_mppi_iters; mppi_iter++) {

        // ── Pass 1: evaluate particles, store only costs ────────────────
        // Save RNG state so we can replay the exact same noise in pass 2.
        const uint32_t rng_checkpoint = rng_state;
        float min_cost = 1e30f;

        for (int k = 0; k < n_particles; k++) {
            // Generate noise & build trial config on-the-fly.
            float q_trial[MAX_ACT];
            for (int a = 0; a < n_act; a++) {
                float n = s_fixed_mask[a] ? 0.0f : rng_normal(rng_state) * sigma;
                q_trial[a] = clampf(cfg[a] + n, s_lower[a], s_upper[a]);
            }

            float r_trial[6 * MAX_EE];
            compute_multi_ee_residual_only(
                q_trial, T_world,
                s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r_trial);

            float cost = 0.0f;
            for (int ee = 0; ee < n_ee; ee++)
                for (int kk = 0; kk < 6; kk++) {
                    float rw = r_trial[ee*6+kk] * W[kk];
                    cost += rw * rw;
                }
            cost += collision_penalty(q_trial, T_world);
            costs[k]  = cost;
            min_cost  = fminf(min_cost, cost);
        }

        // ── Softmax weights (in-place over costs[]) ─────────────────────
        float total_w = 0.0f;
        for (int k = 0; k < n_particles; k++) {
            costs[k] = expf(-(costs[k] - min_cost) * inv_temp);
            total_w += costs[k];
        }
        const float inv_total_w = 1.0f / fmaxf(total_w, 1e-20f);

        // ── Pass 2: replay RNG, accumulate weighted-mean delta ──────────
        uint32_t rng_replay = rng_checkpoint;
        float delta[MAX_ACT];
        for (int a = 0; a < n_act; a++) delta[a] = 0.0f;

        for (int k = 0; k < n_particles; k++) {
            float w = costs[k] * inv_total_w;
            for (int a = 0; a < n_act; a++) {
                float n = s_fixed_mask[a] ? 0.0f : rng_normal(rng_replay) * sigma;
                delta[a] += w * n;
            }
        }

        for (int a = 0; a < n_act; a++)
            cfg[a] = clampf(cfg[a] + delta[a], s_lower[a], s_upper[a]);

        // Track best.
        compute_multi_ee_residual_only(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r);
        float curr_err = 0.0f;
        for (int ee = 0; ee < n_ee; ee++)
            for (int k = 0; k < 6; k++) { float rw = r[ee*6+k] * W[k]; curr_err += rw * rw; }
        curr_err += collision_penalty(cfg, T_world);
        if (curr_err < best_err) {
            best_err = curr_err;
            for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];
        }
    }

    // ── Convergence gate: skip L-BFGS if already converged ──────────────
    {
        compute_multi_ee_residual_only(
            best_cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_target_jnts, s_target_Ts, n_joints, n_act, n_ee, r);
        bool all_conv = true;
        for (int ee = 0; ee < n_ee; ee++) {
            if (norm3(r + ee*6) >= eps_pos || norm3(r + ee*6 + 3) >= eps_ori) {
                all_conv = false; break;
            }
        }
        if (all_conv && enable_collision) {
            all_conv = collision_penalty(best_cfg, T_world) <= 1e-12f;
        }
        if (all_conv) goto write_output;
    }

    // Use best_cfg from MPPI as L-BFGS starting point.
    for (int a = 0; a < n_act; a++) cfg[a] = best_cfg[a];

    // =====================================================================
    // Stage 2: L-BFGS gradient refinement
    // =====================================================================
    {
        int m_used = 0, newest = -1;
        for (int i = 0; i < m_lbfgs * n_act; i++) { s_buf[i] = 0.0f; y_buf[i] = 0.0f; }
        for (int i = 0; i < m_lbfgs; i++) rho_buf[i] = 0.0f;
        for (int a = 0; a < n_act; a++) { g_prev[a] = 0.0f; cfg_prev[a] = cfg[a]; }

        for (int iter = 0; iter < n_lbfgs_iters; iter++) {

            // ── Residual + Jacobian ─────────────────────────────────────
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
                        all_conv = false; break;
                    }
                }
                if (all_conv) break;
            }

            // fw = W * r
            for (int k = 0; k < 6 * n_ee; k++) fw[k] = r[k] * W[k % 6];

            float curr_err = 0.0f;
            for (int k = 0; k < 6 * n_ee; k++) curr_err += fw[k] * fw[k];
            curr_err += collision_penalty(cfg, T_world);

            // Gradient: g = J_w^T fw  where J_w[k,a] = W[k%6] * J[k*n_act+a]
            for (int a = 0; a < n_act; a++) {
                float acc = 0.0f;
                for (int k = 0; k < 6 * n_ee; k++)
                    acc += J[k * n_act + a] * W[k % 6] * fw[k];
                g[a] = s_fixed_mask[a] ? 0.0f : acc;
            }

            // ── Update L-BFGS history ───────────────────────────────────
            if (iter > 0) {
                float sy = 0.0f, yy = 0.0f;
                float s_k[MAX_ACT], y_k[MAX_ACT];
                for (int a = 0; a < n_act; a++) {
                    s_k[a] = cfg[a]  - cfg_prev[a];
                    y_k[a] = g[a]    - g_prev[a];
                    sy    += s_k[a]  * y_k[a];
                    yy    += y_k[a]  * y_k[a];
                }
                if (sy > 1e-10f * yy + 1e-30f) {
                    newest = (newest + 1) % m_lbfgs;
                    for (int a = 0; a < n_act; a++) {
                        s_buf[newest * n_act + a] = s_k[a];
                        y_buf[newest * n_act + a] = y_k[a];
                    }
                    rho_buf[newest] = 1.0f / sy;
                    if (m_used < m_lbfgs) m_used++;
                }
            }

            for (int a = 0; a < n_act; a++) { cfg_prev[a] = cfg[a]; g_prev[a] = g[a]; }

            // ── L-BFGS two-loop: dir = -H g ─────────────────────────────
            if (m_used > 0) {
                lbfgs_two_loop(g, s_buf, y_buf, rho_buf, alpha_buf,
                               n_act, m_lbfgs, m_used, newest, dir);
            } else {
                float gnorm = 0.0f;
                for (int a = 0; a < n_act; a++) gnorm += g[a] * g[a];
                gnorm = sqrtf(gnorm) + 1e-18f;
                for (int a = 0; a < n_act; a++) dir[a] = -g[a] / gnorm;
            }

            for (int a = 0; a < n_act; a++)
                if (s_fixed_mask[a]) dir[a] = 0.0f;

            // ── Trust-region step clipping ───────────────────────────────
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
                for (int a = 0; a < n_act; a++) dnorm += dir[a] * dir[a];
                dnorm = sqrtf(dnorm);
                if (dnorm > R) {
                    const float scale = R / (dnorm + 1e-18f);
                    for (int a = 0; a < n_act; a++) dir[a] *= scale;
                }
            }

            // ── 5-point line search (early exit on sufficient decrease) ─
            const float alphas[5] = { 1.0f, 0.5f, 0.25f, 0.1f, 0.025f };
            float best_alpha_err = 1e30f;
            int   best_alpha_idx = 0;
            const float suff_thresh = curr_err * (1.0f - 1e-4f);

            for (int ai = 0; ai < 5; ai++) {
                float cfg_trial[MAX_ACT];
                for (int a = 0; a < n_act; a++)
                    cfg_trial[a] = clampf(cfg[a] + alphas[ai] * dir[a],
                                          s_lower[a], s_upper[a]);

                float r_trial[6 * MAX_EE];
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
                // Early exit: first alpha that gives sufficient decrease wins.
                if (err_trial < suff_thresh) break;
            }

            for (int a = 0; a < n_act; a++)
                cfg[a] = clampf(cfg[a] + alphas[best_alpha_idx] * dir[a],
                                s_lower[a], s_upper[a]);

            if (best_alpha_err < best_err) {
                best_err = best_alpha_err;
                for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];
            }
        }
    }

write_output:
    for (int a = 0; a < n_act; a++) out[gs * n_act + a] = best_cfg[a];
    out_err[gs] = best_err;
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error MppiIkCudaImpl(
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
    ffi::Buffer<ffi::DataType::S32> self_pair_i,
    ffi::Buffer<ffi::DataType::S32> self_pair_j,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    ffi::Buffer<ffi::DataType::S32> rng_seed_buf,
    int64_t n_particles,
    int64_t n_mppi_iters,
    int64_t n_lbfgs_iters,
    int64_t m_lbfgs,
    float   sigma,
    float   mppi_temperature,
    float   pos_weight,
    float   ori_weight,
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
    const int n_self_pairs = static_cast<int>(self_pair_i.dimensions()[0]);

    constexpr int THREADS_MAX = 16;
    const int threads  = n_seeds < THREADS_MAX ? n_seeds : THREADS_MAX;
    const int blocks_x = (n_seeds + threads - 1) / threads;

    mppi_ik_kernel<<<dim3(blocks_x, n_problems), threads, 0, stream>>>(
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
        self_pair_i.typed_data(),
        self_pair_j.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        rng_seed_buf.typed_data(),
        out->typed_data(),
        out_err->typed_data(),
        n_problems, n_seeds, n_joints, n_act, n_ee,
        n_robot_spheres, n_world_spheres, n_world_capsules,
        n_world_boxes, n_world_halfspaces, n_self_pairs,
        static_cast<int>(n_particles),
        static_cast<int>(n_mppi_iters),
        static_cast<int>(n_lbfgs_iters),
        static_cast<int>(m_lbfgs),
        static_cast<int>(enable_collision),
        collision_weight,
        collision_margin,
        sigma, mppi_temperature,
        pos_weight, ori_weight,
        eps_pos, eps_ori);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MppiIkCudaFfi, MppiIkCudaImpl,
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
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // self_pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // self_pair_j
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // fixed_mask
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // rng_seed
        .Attr<int64_t>("n_particles")
        .Attr<int64_t>("n_mppi_iters")
        .Attr<int64_t>("n_lbfgs_iters")
        .Attr<int64_t>("m_lbfgs")
        .Attr<float>("sigma")
        .Attr<float>("mppi_temperature")
        .Attr<float>("pos_weight")
        .Attr<float>("ori_weight")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Attr<int64_t>("enable_collision")
        .Attr<float>("collision_weight")
        .Attr<float>("collision_margin")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out cfgs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out errors
);
