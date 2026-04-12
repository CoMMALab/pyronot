/**
 * Hit-and-run NLP sampling CUDA kernel.
 *
 * Build with:
 *   bash src/pyronot/cuda_kernels/build_hit_and_run_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstdio>

namespace ffi = xla::ffi;

#ifndef MAX_JOINTS
#define MAX_JOINTS 64
#endif

#ifndef MAX_ACT
#define MAX_ACT 16
#endif

static __device__ __forceinline__
float box_distance_sq(const float* ee, const float* box_min, const float* box_max)
{
    float d = 0.0f;
    for (int k = 0; k < 3; k++) {
        float c = ee[k];
        c = c < box_min[k] ? box_min[k] : (c > box_max[k] ? box_max[k] : c);
        float diff = ee[k] - c;
        d += diff * diff;
    }
    return d;
}

static __device__ void fk_and_residual(
    const float* cfg,
    const float* s_twists,
    const float* s_parent_tf,
    const int*   s_parent_idx,
    const int*   s_act_idx,
    const float* s_mimic_mul,
    const float* s_mimic_off,
    const int*   s_mimic_act_idx,
    const int*   s_topo_inv,
    float*       T_world,
    int          n_joints,
    int          n_act,
    int          target_jnt,
    const float* target_T,
    float*       r)
{
    fk_single(cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
              s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
              T_world, n_joints, n_act);

    const float* T_ee = T_world + target_jnt * 7;
    r[0] = T_ee[4] - target_T[4];
    r[1] = T_ee[5] - target_T[5];
    r[2] = T_ee[6] - target_T[6];

    const float q_ee[4] = { T_ee[0], T_ee[1], T_ee[2], T_ee[3] };
    const float q_tgt[4] = { target_T[0], target_T[1], target_T[2], target_T[3] };
    const float q_tgt_inv[4] = { q_tgt[0], -q_tgt[1], -q_tgt[2], -q_tgt[3] };
    float q_err[4];
    quat_mul(q_ee, q_tgt_inv, q_err);
    if (q_err[0] < 0.0f) {
        q_err[0] = -q_err[0]; q_err[1] = -q_err[1];
        q_err[2] = -q_err[2]; q_err[3] = -q_err[3];
    }
    const float sin_half = sqrtf(q_err[1]*q_err[1] + q_err[2]*q_err[2] + q_err[3]*q_err[3]);
    if (sin_half > 1e-6f) {
        const float theta = 2.0f * atan2f(sin_half, q_err[0]);
        const float inv_sin = theta / sin_half;
        r[3] = q_err[1] * inv_sin;
        r[4] = q_err[2] * inv_sin;
        r[5] = q_err[3] * inv_sin;
    } else {
        r[3] = 2.0f * q_err[1];
        r[4] = 2.0f * q_err[2];
        r[5] = 2.0f * q_err[3];
    }
}

static __device__ void build_jacobian(
    float*       J,
    const float* T_world,
    const float* s_twists,
    const int*   s_act_idx,
    const float* s_mimic_mul,
    const int*   s_mimic_act_idx,
    const int*   s_ancestor_mask,
    int          target_jnt,
    int          n_joints,
    int          n_act)
{
    for (int k = 0; k < 6 * n_act; k++) J[k] = 0.0f;

    const float* ee_pos = T_world + target_jnt * 7 + 4;
    const float arm[3] = { ee_pos[0], ee_pos[1], ee_pos[2] };

    for (int j = 0; j < n_joints; j++) {
        if (!s_ancestor_mask[j]) continue;
        int a1 = s_act_idx[j];
        int a2 = s_mimic_act_idx[j];
        if (a1 < 0 && a2 < 0) continue;

        const float* tw = s_twists + j * 6;
        const float ang_sq = tw[3]*tw[3] + tw[4]*tw[4] + tw[5]*tw[5];
        const float lin_sq = tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2];
        const float* T_j = T_world + j * 7;
        const float* p_j = T_j + 4;

        float jg_lin[3], jg_ang[3];

        if (ang_sq > 1e-6f) {
            const float inv_ang = 1.0f / sqrtf(ang_sq);
            const float body_ax[3] = { tw[3]*inv_ang, tw[4]*inv_ang, tw[5]*inv_ang };
            float z_j[3];
            quat_rotate(T_j, body_ax, z_j);
            const float arm_j[3] = { arm[0]-p_j[0], arm[1]-p_j[1], arm[2]-p_j[2] };
            cross3(z_j, arm_j, jg_lin);
            jg_ang[0] = z_j[0]; jg_ang[1] = z_j[1]; jg_ang[2] = z_j[2];
        } else if (lin_sq > 1e-6f) {
            const float inv_lin = 1.0f / sqrtf(lin_sq);
            const float body_ax[3] = { tw[0]*inv_lin, tw[1]*inv_lin, tw[2]*inv_lin };
            float z_j[3];
            quat_rotate(T_j, body_ax, z_j);
            jg_lin[0] = z_j[0]; jg_lin[1] = z_j[1]; jg_lin[2] = z_j[2];
            jg_ang[0] = 0.0f; jg_ang[1] = 0.0f; jg_ang[2] = 0.0f;
        } else {
            continue;
        }

        if (a1 >= 0) {
            const float ms = s_mimic_mul[j];
            J[0*n_act + a1] += ms * jg_lin[0];
            J[1*n_act + a1] += ms * jg_lin[1];
            J[2*n_act + a1] += ms * jg_lin[2];
            J[3*n_act + a1] += ms * jg_ang[0];
            J[4*n_act + a1] += ms * jg_ang[1];
            J[5*n_act + a1] += ms * jg_ang[2];
        }
        if (a2 >= 0) {
            const float ms = s_mimic_mul[j];
            J[0*n_act + a2] += ms * jg_lin[0];
            J[1*n_act + a2] += ms * jg_lin[1];
            J[2*n_act + a2] += ms * jg_lin[2];
            J[3*n_act + a2] += ms * jg_ang[0];
            J[4*n_act + a2] += ms * jg_ang[1];
            J[5*n_act + a2] += ms * jg_ang[2];
        }
    }
}

static __device__ void lm_step(
    float*       cfg,
    float*       J,
    const float* r,
    const float  W[6],
    float*       col_scale,
    float*       A,
    float*       rhs,
    const float* s_lower,
    const float* s_upper,
    const int*   s_fixed_mask,
    int          n_act,
    float        lam)
{
    for (int a = 0; a < n_act; a++) {
        float sq = 0.0f;
        for (int k = 0; k < 6; k++) sq += (W[k] * J[k*n_act+a]) * (W[k] * J[k*n_act+a]);
        col_scale[a] = sqrtf(sq) + 1e-8f;
    }

    for (int i = 0; i < n_act; i++) {
        for (int j = 0; j < n_act; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 6; k++)
                acc += (W[k]*J[k*n_act+i]) * (W[k]*J[k*n_act+j]) / (col_scale[i] * col_scale[j]);
            A[i*n_act + j] = acc;
            if (i == j) A[i*n_act + j] += lam;
        }
        float rb = 0.0f;
        for (int k = 0; k < 6; k++)
            rb += (W[k]*J[k*n_act+i]) * r[k] / col_scale[i];
        rhs[i] = -rb;
    }

    for (int a = 0; a < n_act; a++) {
        if (!s_fixed_mask[a]) continue;
        for (int j = 0; j < n_act; j++) A[a*n_act+j] = A[j*n_act+a] = 0.0f;
        A[a*n_act+a] = 1.0f;
        rhs[a] = 0.0f;
    }

    chol_solve(A, rhs, n_act);

    float delta[MAX_ACT];
    for (int a = 0; a < n_act; a++) delta[a] = rhs[a] / col_scale[a];

    float dnorm = 0.0f;
    for (int a = 0; a < n_act; a++) dnorm += delta[a] * delta[a];
    dnorm = sqrtf(dnorm);

    const float R = 0.18f;
    const float scale = (dnorm > R) ? R / (dnorm + 1e-18f) : 1.0f;

    for (int a = 0; a < n_act; a++)
        cfg[a] = clampf(cfg[a] + scale * delta[a], s_lower[a], s_upper[a]);
}

__global__
void hit_and_run_ik_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ ancestor_masks,
    const float* __restrict__ box_mins,   // (n_problems * 3) — per-problem box min
    const float* __restrict__ box_maxs,   // (n_problems * 3) — per-problem box max
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    const int*   __restrict__ rng_seed_ptr,
    float*       __restrict__ out_cfg,
    float*       __restrict__ out_err,
    float*       __restrict__ out_ee_points,
    float*       __restrict__ out_targets,
    int n_problems, int n_samples, int n_joints, int n_act, int target_jnt,
    int max_iter, int n_iterations,
    float pos_weight, float ori_weight, float lambda_init,
    float eps_pos, float eps_ori, float noise_std)
{
    __shared__ float s_twists[MAX_JOINTS * 6];
    __shared__ float s_parent_tf[MAX_JOINTS * 7];
    __shared__ int   s_parent_idx[MAX_JOINTS];
    __shared__ int   s_act_idx[MAX_JOINTS];
    __shared__ float s_mimic_mul[MAX_JOINTS];
    __shared__ float s_mimic_off[MAX_JOINTS];
    __shared__ int   s_mimic_act_idx[MAX_JOINTS];
    __shared__ int   s_topo_inv[MAX_JOINTS];
    __shared__ int   s_ancestor_mask[MAX_JOINTS];
    __shared__ float s_lower[MAX_ACT];
    __shared__ float s_upper[MAX_ACT];
    __shared__ int   s_fixed_mask[MAX_ACT];
    // NOTE: box bounds are per-problem; each warp reads its own bounds into
    // local registers (w_box_min/w_box_max) rather than block-wide shared memory.

    for (int i = threadIdx.x; i < n_joints * 6; i += blockDim.x) s_twists[i]    = twists[i];
    for (int i = threadIdx.x; i < n_joints * 7; i += blockDim.x) s_parent_tf[i] = parent_tf[i];
    for (int i = threadIdx.x; i < n_joints; i += blockDim.x) {
        s_parent_idx[i]    = parent_idx[i];
        s_act_idx[i]       = act_idx[i];
        s_mimic_mul[i]     = mimic_mul[i];
        s_mimic_off[i]     = mimic_off[i];
        s_mimic_act_idx[i] = mimic_act_idx[i];
        s_topo_inv[i]      = topo_inv[i];
        s_ancestor_mask[i] = ancestor_masks[i];
    }
    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[i];
        s_upper[i]      = upper[i];
        s_fixed_mask[i] = fixed_mask[i];
    }
    __syncthreads();

    const int warp_id_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;

    const int global_chain_idx = blockIdx.x * warps_per_block + warp_id_in_block;
    if (global_chain_idx >= n_problems * n_samples) return;

    const int problem_idx = global_chain_idx / n_samples;
    const int chain_idx = global_chain_idx % n_samples;

    // Per-warp box bounds: lane 0 caches the 6 floats from global memory.
    // Other lanes never access these; only lane 0 uses them for target generation.
    float w_box_min[3], w_box_max[3];
    if (lane_id == 0) {
        for (int k = 0; k < 3; k++) {
            w_box_min[k] = box_mins[problem_idx * 3 + k];
            w_box_max[k] = box_maxs[problem_idx * 3 + k];
        }
    }

    uint32_t rng_state = (uint32_t)(*rng_seed_ptr)
                       ^ (uint32_t)(chain_idx * 0x9e3779b9u)
                       ^ (uint32_t)(problem_idx * 0x6c62272eu);
    xorshift32(rng_state); xorshift32(rng_state); xorshift32(rng_state);

    extern __shared__ float smem[];
    float* ws = smem + warp_id_in_block * (MAX_JOINTS * 7 + MAX_ACT * 6 + MAX_ACT + MAX_ACT + MAX_ACT + 6 + MAX_ACT + MAX_ACT * MAX_ACT + MAX_ACT + 7);

    float* T_world = ws;
    float* J       = T_world + MAX_JOINTS * 7;
    float* cfg     = J       + MAX_ACT * 6;
    float* best_cfg = cfg    + MAX_ACT;
    float* r       = best_cfg + MAX_ACT;
    float* col_scale = r + 6;
    float* A         = col_scale + MAX_ACT;
    float* rhs       = A + MAX_ACT * MAX_ACT;
    float* direction = rhs + MAX_ACT;           // n_act floats: shared direction for all lanes
    float* s_target  = direction + MAX_ACT;     // 7 floats: shared target pose for all lanes

    const float W[6] = {
        pos_weight, pos_weight, pos_weight,
        ori_weight, ori_weight, ori_weight,
    };
    (void)noise_std;
    float lam = lambda_init;

    for (int a = lane_id; a < n_act; a += 32)
        cfg[a] = best_cfg[a] = seeds[global_chain_idx * n_act + a];
    __syncwarp();

    float best_err = 1e30f;

    for (int iter = 0; iter < n_iterations; iter++) {
        lam = lambda_init;

        if (iter > 0) {
            // Hit-and-run step from current feasible state:
            // 1) sample random direction on S^(n-1),
            // 2) clip with joint limits to get [lambda_min, lambda_max],
            // 3) sample lambda uniformly on the chord.
            float local_norm_sq = 0.0f;
            for (int a = lane_id; a < n_act; a += 32) {
                float g = 0.0f;
                if (!s_fixed_mask[a]) {
                    g = rng_normal(rng_state);
                    local_norm_sq += g * g;
                }
                direction[a] = g;
            }

            for (int off = 16; off > 0; off >>= 1) {
                local_norm_sq += __shfl_down_sync(0xffffffffu, local_norm_sq, off);
            }

            float inv_dir_norm = 0.0f;
            if (lane_id == 0) {
                const float norm_sq = fmaxf(local_norm_sq, 1e-20f);
                inv_dir_norm = rsqrtf(norm_sq);
            }
            inv_dir_norm = __shfl_sync(0xffffffffu, inv_dir_norm, 0);

            for (int a = lane_id; a < n_act; a += 32) {
                direction[a] *= inv_dir_norm;
            }
            __syncwarp();

            float local_lmin = -1e30f;
            float local_lmax =  1e30f;
            for (int a = lane_id; a < n_act; a += 32) {
                if (s_fixed_mask[a]) continue;
                const float d = direction[a];
                if (fabsf(d) <= 1e-9f) continue;

                float lo = (s_lower[a] - best_cfg[a]) / d;
                float hi = (s_upper[a] - best_cfg[a]) / d;
                if (lo > hi) {
                    const float tmp = lo;
                    lo = hi;
                    hi = tmp;
                }
                local_lmin = fmaxf(local_lmin, lo);
                local_lmax = fminf(local_lmax, hi);
            }

            for (int off = 16; off > 0; off >>= 1) {
                local_lmin = fmaxf(local_lmin, __shfl_down_sync(0xffffffffu, local_lmin, off));
                local_lmax = fminf(local_lmax, __shfl_down_sync(0xffffffffu, local_lmax, off));
            }

            float sampled_lambda = 0.0f;
            int valid_interval = 1;
            if (lane_id == 0) {
                const float lambda_min = local_lmin;
                const float lambda_max = local_lmax;
                if (!(lambda_min < lambda_max)) {
                    valid_interval = 0;
                    sampled_lambda = 0.0f;
                } else {
                    const float u = (float)(xorshift32(rng_state) >> 8) * (1.0f / 16777216.0f);
                    sampled_lambda = lambda_min + u * (lambda_max - lambda_min);
                }
            }
            sampled_lambda = __shfl_sync(0xffffffffu, sampled_lambda, 0);
            valid_interval = __shfl_sync(0xffffffffu, valid_interval, 0);

            for (int a = lane_id; a < n_act; a += 32) {
                if (s_fixed_mask[a] || !valid_interval) {
                    cfg[a] = best_cfg[a];
                } else {
                    cfg[a] = clampf(best_cfg[a] + sampled_lambda * direction[a], s_lower[a], s_upper[a]);
                }
            }
            __syncwarp();
            best_err = 1e30f;
        }

        // Sample a random Cartesian target inside the box.
        // Only lane 0 writes into the per-warp shared workspace so that all
        // lanes see a consistent target_T after __syncwarp().
        if (lane_id == 0) {
            const float u0 = (float)(xorshift32(rng_state) >> 8) * (1.0f / 16777216.0f);
            const float u1 = (float)(xorshift32(rng_state) >> 8) * (1.0f / 16777216.0f);
            const float u2 = (float)(xorshift32(rng_state) >> 8) * (1.0f / 16777216.0f);
            s_target[0] = 1.0f; s_target[1] = 0.0f; s_target[2] = 0.0f; s_target[3] = 0.0f;
            s_target[4] = w_box_min[0] + u0 * (w_box_max[0] - w_box_min[0]);
            s_target[5] = w_box_min[1] + u1 * (w_box_max[1] - w_box_min[1]);
            s_target[6] = w_box_min[2] + u2 * (w_box_max[2] - w_box_min[2]);
        }
        __syncwarp();

        for (int lm_iter = 0; lm_iter < max_iter; lm_iter++) {
            fk_and_residual(cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                            T_world, n_joints, n_act, target_jnt, s_target, r);

            float curr_err = 0.0f;
            for (int k = 0; k < 6; k++) {
                float rw = r[k] * W[k];
                curr_err += rw * rw;
            }

            if (curr_err < best_err) {
                best_err = curr_err;
                for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];
            }

            if (norm3(r) < eps_pos && norm3(r+3) < eps_ori) break;

            build_jacobian(J, T_world, s_twists, s_act_idx, s_mimic_mul,
                          s_mimic_act_idx, s_ancestor_mask, target_jnt, n_joints, n_act);

            lm_step(cfg, J, r, W, col_scale, A, rhs, s_lower, s_upper, s_fixed_mask,
                   n_act, lam);

            float r_t[6];
            fk_and_residual(cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                            T_world, n_joints, n_act, target_jnt, s_target, r_t);

            float trial_err = 0.0f;
            for (int k = 0; k < 6; k++) {
                float rw = r_t[k] * W[k];
                trial_err += rw * rw;
            }

            const bool improved = trial_err < curr_err * (1.0f - 1e-4f);
            if (improved) lam = fmaxf(lam * 0.5f, 1e-10f);
            else lam = fminf(lam * 3.0f, 1e6f);

            if (trial_err < best_err) {
                best_err = trial_err;
                for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];
            }
        }
    }

    const int final_idx = problem_idx * n_samples + chain_idx;
    for (int a = lane_id; a < n_act; a += 32)
        out_cfg[final_idx * n_act + a] = best_cfg[a];

    if (lane_id == 0) {
        out_err[final_idx] = best_err;
        fk_single(best_cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                  s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                  T_world, n_joints, n_act);
        const float* T_ee_final = T_world + target_jnt * 7;
        out_ee_points[final_idx * 3 + 0] = T_ee_final[4];
        out_ee_points[final_idx * 3 + 1] = T_ee_final[5];
        out_ee_points[final_idx * 3 + 2] = T_ee_final[6];
        out_targets[final_idx * 3 + 0] = s_target[4];
        out_targets[final_idx * 3 + 1] = s_target[5];
        out_targets[final_idx * 3 + 2] = s_target[6];
    }
}

static ffi::Error HitAndRunIkCudaImpl(
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
    ffi::Buffer<ffi::DataType::S32> ancestor_masks,
    ffi::Buffer<ffi::DataType::F32> box_mins,   // (n_problems, 3) — per-problem
    ffi::Buffer<ffi::DataType::F32> box_maxs,   // (n_problems, 3) — per-problem
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    ffi::Buffer<ffi::DataType::S32> rng_seed,
    int64_t target_jnt,
    int64_t max_iter,
    int64_t n_iterations,
    float   pos_weight,
    float   ori_weight,
    float   lambda_init,
    float   eps_pos,
    float   eps_ori,
    float   noise_std,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_cfg,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_ee_points,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_targets)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_samples  = static_cast<int>(seeds.dimensions()[1]);
    const int n_act      = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);

    if (n_act > MAX_ACT || n_joints > MAX_JOINTS) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "HitAndRunIkCuda: compile-time limits exceeded (MAX_ACT/MAX_JOINTS)."
        );
    }

    const int tpb = 128;
    if (tpb < 32 || tpb > 1024 || tpb % 32 != 0) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "HitAndRunIkCuda: threads_per_block must be a multiple of 32 in [32, 1024]."
        );
    }

    const int warps_per_block = tpb / 32;
    const int total_warps     = n_problems * n_samples;
    const int blocks          = (total_warps + warps_per_block - 1) / warps_per_block;

    const size_t smem_per_warp = (MAX_JOINTS * 7 + MAX_ACT * 6 + MAX_ACT + MAX_ACT + MAX_ACT + 6 + MAX_ACT + MAX_ACT * MAX_ACT + MAX_ACT + 7) * sizeof(float);
    const size_t smem_bytes    = static_cast<size_t>(warps_per_block) * smem_per_warp;

    hit_and_run_ik_kernel<<<dim3(blocks), tpb, smem_bytes, stream>>>(
        seeds.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        ancestor_masks.typed_data(),
        box_mins.typed_data(),
        box_maxs.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        rng_seed.typed_data(),
        out_cfg->typed_data(),
        out_err->typed_data(),
        out_ee_points->typed_data(),
        out_targets->typed_data(),
        n_problems, n_samples, n_joints, n_act,
        static_cast<int>(target_jnt),
        static_cast<int>(max_iter),
        static_cast<int>(n_iterations),
        pos_weight, ori_weight, lambda_init,
        eps_pos, eps_ori, noise_std);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    HitAndRunIkCudaFfi,
    HitAndRunIkCudaImpl,
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
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // ancestor_masks
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // box_mins  (n_problems, 3)
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // box_maxs  (n_problems, 3)
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // fixed_mask
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // rng_seed
        .Attr<int64_t>("target_jnt")
        .Attr<int64_t>("max_iter")
        .Attr<int64_t>("n_iterations")
        .Attr<float>("pos_weight")
        .Attr<float>("ori_weight")
        .Attr<float>("lambda_init")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Attr<float>("noise_std")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_cfg
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_err
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_ee_points
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_targets
);
