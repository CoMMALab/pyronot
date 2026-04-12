/**
 * Brownian-motion region IK CUDA kernel: Warp-per-Problem optimised.
 *
 * Key changes from the Thread-per-Problem version:
 *
 *   1. Warp-per-Problem assignment: one 32-thread warp handles each
 *      (problem, seed) pair.  All control-flow decisions (box-distance
 *      check, corrective-GN trigger) are based on shared scalars in
 *      per-warp shared memory, so all 32 lanes always take the same
 *      branch → zero intra-warp divergence.
 *
 *   2. Shared-memory workspace: T_world (MAX_JOINTS×7), J (6×MAX_ACT),
 *      A (MAX_ACT²), cfg, best_cfg, r, col_scale, rhs are held in a
 *      per-warp region of dynamic shared memory.  This removes the
 *      ~3 KB of local-memory (register-spill) pressure per thread that
 *      caused L1 thrashing in the original kernel.
 *
 *   3. Parallel Jacobian and GN normal-equations build: the n_joints
 *      loop (Jacobian) and the n_act² loop (A matrix) are distributed
 *      across the 32 warp lanes; __shfl_down_sync warp-reductions handle
 *      the trust-region norm and the J_pos @ ξ dot products.
 *
 *   4. Analytical 3×3 solver: Phase-2 null-space projection uses a
 *      closed-form Cramer's-rule solve for the 3×3 system, replacing the
 *      general looped Cholesky.  No loops, no branches.
 *
 *   5. Batched-box support: each problem carries its own axis-aligned
 *      box defined by box_mins[p*3 .. p*3+2] and box_maxs[p*3 .. p*3+2].
 *      This enables multiple distinct regions to be solved in a single
 *      kernel launch with zero overhead.
 *
 *   6. Build with --use_fast_math for hardware SFU (sqrtf, rsqrtf, etc.).
 *
 * Build:
 *   bash src/pyronot/cuda_kernels/build_brownian_motion_ik_cuda.sh
 */

#include "_ik_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstdio>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Per-warp shared-memory layout (in floats)
// ---------------------------------------------------------------------------

#define WSMEM_T_WORLD_SZ  (MAX_JOINTS * 7)
#define WSMEM_J_SZ        (6 * MAX_ACT)
#define WSMEM_CFG_SZ      MAX_ACT
#define WSMEM_BCFG_SZ     MAX_ACT
#define WSMEM_R_SZ        6
#define WSMEM_CSCALE_SZ   MAX_ACT
#define WSMEM_RHS_SZ      MAX_ACT
#define WSMEM_A_SZ        (MAX_ACT * MAX_ACT)
#define WSMEM_SCRATCH_SZ  32     // reductions + misc scalars

// scratch layout (float indices):
//   [0]      = best_err
//   [1]      = current d (box distance)
//   [2]      = norm3(r) (Phase 1 convergence check)
//   [3..5]   = v3 = J_pos @ ξ, then w = B3^{-1} v3 (Phase 2 null-space)
//   [6..8]   = corr_target (Phase 2 corrective GN)

#define WSMEM_PER_WARP_F  (WSMEM_T_WORLD_SZ + WSMEM_J_SZ \
                           + WSMEM_CFG_SZ + WSMEM_BCFG_SZ \
                           + WSMEM_R_SZ + WSMEM_CSCALE_SZ \
                           + WSMEM_RHS_SZ + WSMEM_A_SZ \
                           + WSMEM_SCRATCH_SZ)

// Static shared memory used by per-block cached kinematics arrays.
// Box bounds are per-problem and held in registers on lane 0, so they
// are NOT included in this budget.
#define STATIC_SMEM_BYTES ( \
    sizeof(float) * ( \
        (MAX_JOINTS * 6) + (MAX_JOINTS * 7) + (MAX_JOINTS) + (MAX_JOINTS) + \
        (MAX_ACT) + (MAX_ACT) + 4 \
    ) + \
    sizeof(int) * ( \
        (MAX_JOINTS) + (MAX_JOINTS) + (MAX_JOINTS) + (MAX_JOINTS) + (MAX_JOINTS) + (MAX_ACT) \
    ) \
)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Squared distance from ee[3] to nearest point in axis-aligned box.
static __device__ __forceinline__
float box_dist_sq(const float* __restrict__ ee,
                  const float* __restrict__ box_min,
                  const float* __restrict__ box_max)
{
    float d = 0.0f;
    for (int k = 0; k < 3; k++) {
        const float c    = clampf(ee[k], box_min[k], box_max[k]);
        const float diff = ee[k] - c;
        d += diff * diff;
    }
    return d;
}

/**
 * Analytical solve for a 3×3 symmetric positive-definite system A x = b.
 * A3 is stored row-major with stride 3.  b is overwritten with the solution.
 * Uses Cramer's rule: no loops, no branches.
 */
static __device__ __forceinline__
void solve3x3_sym(float* __restrict__ A3, float* __restrict__ b)
{
    const float a00 = A3[0], a01 = A3[1], a02 = A3[2];
    const float              a11 = A3[4], a12 = A3[5];
    const float                           a22 = A3[8];
    // symmetry: a10=a01, a20=a02, a21=a12

    const float det = a00*(a11*a22 - a12*a12)
                    - a01*(a01*a22 - a12*a02)
                    + a02*(a01*a12 - a11*a02);
    const float inv_det = 1.0f / (det + 1e-12f);

    const float b0 = b[0], b1 = b[1], b2 = b[2];
    b[0] = inv_det * ((a11*a22 - a12*a12)*b0
                    + (a12*a02 - a01*a22)*b1
                    + (a01*a12 - a11*a02)*b2);
    b[1] = inv_det * ((a12*a02 - a01*a22)*b0
                    + (a00*a22 - a02*a02)*b1
                    + (a01*a02 - a00*a12)*b2);
    b[2] = inv_det * ((a01*a12 - a11*a02)*b0
                    + (a01*a02 - a00*a12)*b1
                    + (a00*a11 - a01*a01)*b2);
}

/**
 * Compute IK residual r[6] from an already-computed T_world.
 * Called on lane 0 only (6 components; not worth parallelising).
 */
static __device__ __forceinline__
void compute_r_from_T_world(
    const float* __restrict__ T_world,
    int target_jnt,
    const float* __restrict__ target_T,
    float* __restrict__ r)
{
    const float* T_ee = T_world + target_jnt * 7;
    r[0] = T_ee[4] - target_T[4];
    r[1] = T_ee[5] - target_T[5];
    r[2] = T_ee[6] - target_T[6];

    const float q_tgt_inv[4] = { target_T[0], -target_T[1], -target_T[2], -target_T[3] };
    float q_err[4];
    quat_mul(T_ee, q_tgt_inv, q_err);
    if (q_err[0] < 0.0f) {
        q_err[0] = -q_err[0]; q_err[1] = -q_err[1];
        q_err[2] = -q_err[2]; q_err[3] = -q_err[3];
    }
    const float sin_half = sqrtf(q_err[1]*q_err[1] + q_err[2]*q_err[2] + q_err[3]*q_err[3]);
    if (sin_half > 1e-6f) {
        const float theta   = 2.0f * atan2f(sin_half, q_err[0]);
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

/**
 * Warp-parallel Jacobian build.  ALL 32 lanes must call this together.
 *
 * Clears w_J, then each lane handles joints {lane, lane+32, …}.
 * Contributions to the same column are accumulated with atomicAdd (safe for
 * shared memory; only arises for mimic joints mapped to the same actuator).
 */
static __device__ void build_jacobian_warp(
    float* __restrict__       w_J,
    const float* __restrict__ w_T_world,
    const float* __restrict__ s_twists,
    const int*   __restrict__ s_act_idx,
    const float* __restrict__ s_mimic_mul,
    const int*   __restrict__ s_mimic_act_idx,
    const int*   __restrict__ s_ancestor_mask,
    int n_joints, int n_act, int target_jnt, int lane)
{
    // Zero J in parallel.
    for (int i = lane; i < 6 * n_act; i += 32) w_J[i] = 0.0f;
    __syncwarp();

    const float* ee_pos = w_T_world + target_jnt * 7 + 4;

    for (int j = lane; j < n_joints; j += 32) {
        if (!s_ancestor_mask[j]) continue;
        const int a1 = s_act_idx[j];
        const int a2 = s_mimic_act_idx[j];
        if (a1 < 0 && a2 < 0) continue;

        const float* tw     = s_twists + j * 6;
        const float  ang_sq = tw[3]*tw[3] + tw[4]*tw[4] + tw[5]*tw[5];
        const float  lin_sq = tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2];
        const float* T_j    = w_T_world + j * 7;

        float jg_lin[3], jg_ang[3];

        if (ang_sq > 1e-6f) {
            const float inv_ang    = rsqrtf(ang_sq);
            const float body_ax[3] = { tw[3]*inv_ang, tw[4]*inv_ang, tw[5]*inv_ang };
            float z_j[3];
            quat_rotate(T_j, body_ax, z_j);
            const float arm_j[3] = { ee_pos[0]-T_j[4], ee_pos[1]-T_j[5], ee_pos[2]-T_j[6] };
            cross3(z_j, arm_j, jg_lin);
            jg_ang[0] = z_j[0]; jg_ang[1] = z_j[1]; jg_ang[2] = z_j[2];
        } else if (lin_sq > 1e-6f) {
            const float inv_lin    = rsqrtf(lin_sq);
            const float body_ax[3] = { tw[0]*inv_lin, tw[1]*inv_lin, tw[2]*inv_lin };
            float z_j[3];
            quat_rotate(T_j, body_ax, z_j);
            jg_lin[0] = z_j[0]; jg_lin[1] = z_j[1]; jg_lin[2] = z_j[2];
            jg_ang[0] = 0.0f;   jg_ang[1] = 0.0f;   jg_ang[2] = 0.0f;
        } else {
            continue;
        }

        const float ms = s_mimic_mul[j];
        if (a1 >= 0) {
            atomicAdd(&w_J[0*n_act + a1], ms * jg_lin[0]);
            atomicAdd(&w_J[1*n_act + a1], ms * jg_lin[1]);
            atomicAdd(&w_J[2*n_act + a1], ms * jg_lin[2]);
            atomicAdd(&w_J[3*n_act + a1], ms * jg_ang[0]);
            atomicAdd(&w_J[4*n_act + a1], ms * jg_ang[1]);
            atomicAdd(&w_J[5*n_act + a1], ms * jg_ang[2]);
        }
        if (a2 >= 0) {
            atomicAdd(&w_J[0*n_act + a2], ms * jg_lin[0]);
            atomicAdd(&w_J[1*n_act + a2], ms * jg_lin[1]);
            atomicAdd(&w_J[2*n_act + a2], ms * jg_lin[2]);
            atomicAdd(&w_J[3*n_act + a2], ms * jg_ang[0]);
            atomicAdd(&w_J[4*n_act + a2], ms * jg_ang[1]);
            atomicAdd(&w_J[5*n_act + a2], ms * jg_ang[2]);
        }
    }
    __syncwarp();
}

/**
 * Warp-parallel Gauss-Newton step.  ALL 32 lanes must call this together.
 *
 * Builds (J_WD^T J_WD + λI) and rhs in parallel, pins fixed joints, then
 * calls chol_solve on lane 0.  A warp-reduction computes the trust-region
 * norm; the scaled step is applied in parallel.  Trust-region radius = 0.18.
 *
 * w_A and w_rhs are scratch and are overwritten.
 */
static __device__ void gn_step_warp(
    float* __restrict__       w_J,
    float* __restrict__       w_r,
    float* __restrict__       w_cfg,
    float* __restrict__       w_col_scale,
    float* __restrict__       w_A,
    float* __restrict__       w_rhs,
    const float* __restrict__ s_lower,
    const float* __restrict__ s_upper,
    const int*   __restrict__ s_fixed_mask,
    const float W[6],
    int n_act, float lam, int lane)
{
    // 1. col_scale[a] = ||W .* J[:,a]||_2 + eps  (parallel over n_act)
    for (int a = lane; a < n_act; a += 32) {
        float sq = 0.0f;
        for (int k = 0; k < 6; k++) {
            const float wj = W[k] * w_J[k*n_act + a];
            sq += wj * wj;
        }
        w_col_scale[a] = sqrtf(sq) + 1e-8f;
    }
    __syncwarp();

    // 2. Build A (n_act × n_act) and rhs (n_act) in parallel.
    const int n_act2 = n_act * n_act;
    for (int ij = lane; ij < n_act2; ij += 32) {
        const int i = ij / n_act, jj = ij % n_act;
        float acc = 0.0f;
        for (int k = 0; k < 6; k++)
            acc += W[k]*W[k] * w_J[k*n_act+i] * w_J[k*n_act+jj];
        w_A[ij] = acc / (w_col_scale[i] * w_col_scale[jj]);
        if (i == jj) w_A[ij] += lam;
    }
    for (int i = lane; i < n_act; i += 32) {
        float rb = 0.0f;
        for (int k = 0; k < 6; k++)
            rb += W[k]*W[k] * w_J[k*n_act+i] * w_r[k];
        w_rhs[i] = -rb / w_col_scale[i];
    }
    __syncwarp();

    // 3. Pin fixed joints (parallel; benign multi-write of 0 is safe).
    for (int a = lane; a < n_act; a += 32) {
        if (!s_fixed_mask[a]) continue;
        for (int jj = 0; jj < n_act; jj++) {
            w_A[a*n_act + jj] = 0.0f;
            w_A[jj*n_act + a] = 0.0f;
        }
        w_A[a*n_act + a] = 1.0f;
        w_rhs[a]          = 0.0f;
    }
    __syncwarp();

    // 4. Cholesky solve (serial, lane 0; n_act is small).
    if (lane == 0) chol_solve(w_A, w_rhs, n_act);
    __syncwarp();

    // 5. Trust-region norm via warp reduction.
    float local_dsq = 0.0f;
    for (int a = lane; a < n_act; a += 32) {
        const float da = w_rhs[a] / w_col_scale[a];
        local_dsq += da * da;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_dsq += __shfl_down_sync(0xffffffff, local_dsq, off);
    const float dnorm = sqrtf(__shfl_sync(0xffffffff, local_dsq, 0));

    const float R  = 0.18f;
    const float sc = (dnorm > R) ? R / (dnorm + 1e-18f) : 1.0f;

    // 6. Apply scaled step (parallel over n_act).
    for (int a = lane; a < n_act; a += 32)
        w_cfg[a] = clampf(w_cfg[a] + sc * w_rhs[a] / w_col_scale[a],
                          s_lower[a], s_upper[a]);
    __syncwarp();
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__
void brownian_motion_ik_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ init_points,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ ancestor_mask,
    const float* __restrict__ target_quat,
    const float* __restrict__ box_mins,   // (n_problems * 3) — per-problem box min
    const float* __restrict__ box_maxs,   // (n_problems * 3) — per-problem box max
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    const int*   __restrict__ rng_seed_ptr,
    float*       __restrict__ out_cfg,
    float*       __restrict__ out_err,
    float*       __restrict__ out_ee_points,
    float*       __restrict__ out_target_points,
    int n_problems, int n_seeds, int n_joints, int n_act,
    int target_jnt, int max_iter,
    float pos_weight, float ori_weight, float lambda_init, float eps_pos,
    float noise_std, int n_brownian_steps, int fk_check_freq)
{
    // ── Static kinematics in shared memory (shared across the whole block) ──
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
    __shared__ float s_target_quat[4];
    // NOTE: box bounds are per-problem and are held in lane-0 local registers
    // (w_box_min/w_box_max), not in block-wide shared memory.

    for (int i = threadIdx.x; i < n_joints * 6; i += blockDim.x) s_twists[i]      = twists[i];
    for (int i = threadIdx.x; i < n_joints * 7; i += blockDim.x) s_parent_tf[i]   = parent_tf[i];
    for (int i = threadIdx.x; i < n_joints; i += blockDim.x) {
        s_parent_idx[i]    = parent_idx[i];
        s_act_idx[i]       = act_idx[i];
        s_mimic_mul[i]     = mimic_mul[i];
        s_mimic_off[i]     = mimic_off[i];
        s_mimic_act_idx[i] = mimic_act_idx[i];
        s_topo_inv[i]      = topo_inv[i];
        s_ancestor_mask[i] = ancestor_mask[i];
    }
    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[i];
        s_upper[i]      = upper[i];
        s_fixed_mask[i] = fixed_mask[i];
    }
    for (int i = threadIdx.x; i < 4; i += blockDim.x) s_target_quat[i] = target_quat[i];
    __syncthreads();

    // ── Per-warp dynamic shared-memory workspace ────────────────────────────
    extern __shared__ float wsmem_raw[];
    const int warp_in_block  = threadIdx.x >> 5;
    const int lane            = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;

    float* ws          = wsmem_raw + warp_in_block * WSMEM_PER_WARP_F;
    float* w_T_world   = ws;
    float* w_J         = w_T_world + WSMEM_T_WORLD_SZ;
    float* w_cfg       = w_J       + WSMEM_J_SZ;
    float* w_best_cfg  = w_cfg     + WSMEM_CFG_SZ;
    float* w_r         = w_best_cfg+ WSMEM_BCFG_SZ;
    float* w_col_scale = w_r       + WSMEM_R_SZ;
    float* w_rhs       = w_col_scale + WSMEM_CSCALE_SZ;
    float* w_A         = w_rhs     + WSMEM_RHS_SZ;
    float* w_scratch   = w_A       + WSMEM_A_SZ;

    // ── Warp / thread identity ──────────────────────────────────────────────
    const int warp_global_id = blockIdx.x * warps_per_block + warp_in_block;
    if (warp_global_id >= n_problems * n_seeds) return;
    const int p = warp_global_id / n_seeds;

    // ── Per-warp box bounds (lane 0 reads from per-problem global arrays) ───
    // Only lane 0 uses these; other lanes never access them.
    float w_box_min[3], w_box_max[3];
    if (lane == 0) {
        for (int k = 0; k < 3; k++) {
            w_box_min[k] = box_mins[p * 3 + k];
            w_box_max[k] = box_maxs[p * 3 + k];
        }
    }

    // ── Initialise per-warp workspace ──────────────────────────────────────
    for (int a = lane; a < n_act; a += 32) {
        const float val = seeds[warp_global_id * n_act + a];
        w_cfg[a]      = val;
        w_best_cfg[a] = val;
    }
    if (lane == 0) w_scratch[0] = 1e30f;   // best_err
    __syncwarp();

    // Per-lane RNG: each lane has a unique initial state for diverse noise.
    uint32_t rng_state = (uint32_t)(*rng_seed_ptr)
                       ^ (uint32_t)((warp_global_id % n_seeds) * 0x9e3779b9u + (uint32_t)lane * 0x6c62272eu)
                       ^ (uint32_t)(p * 0x6c62272eu + (uint32_t)lane * 0x9e3779b9u);
    xorshift32(rng_state); xorshift32(rng_state); xorshift32(rng_state);

    const float target_p[3] = {
        init_points[warp_global_id * 3 + 0],
        init_points[warp_global_id * 3 + 1],
        init_points[warp_global_id * 3 + 2],
    };
    const float W[6] = {
        pos_weight, pos_weight, pos_weight,
        ori_weight, ori_weight, ori_weight,
    };
    const float lam = lambda_init;

    // ─── Phase 1: GN Boundary Reach ────────────────────────────────────────
    for (int iter = 0; iter < max_iter; iter++) {

        // FK (serial, lane 0 → writes w_T_world in shared memory).
        if (lane == 0) {
            fk_single(w_cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                      s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                      w_T_world, n_joints, n_act);
        }
        __syncwarp();

        // Residual + box distance (serial, lane 0; few ops).
        if (lane == 0) {
            float target_T[7] = {
                s_target_quat[0], s_target_quat[1], s_target_quat[2], s_target_quat[3],
                target_p[0], target_p[1], target_p[2],
            };
            compute_r_from_T_world(w_T_world, target_jnt, target_T, w_r);
            const float* ee = w_T_world + target_jnt * 7 + 4;
            const float  d  = box_dist_sq(ee, w_box_min, w_box_max);
            w_scratch[1] = d;
            w_scratch[2] = norm3(w_r);
            if (d < w_scratch[0]) {
                w_scratch[0] = d;
                for (int a = 0; a < n_act; a++) w_best_cfg[a] = w_cfg[a];
            }
        }
        __syncwarp();

        // Early exits: all lanes read the same shared scalars → no divergence.
        if (w_scratch[1] == 0.0f) break;
        if (w_scratch[2] < eps_pos) break;

        // Jacobian: recompute every 3rd iteration (warp-parallel).
        if (iter % 3 == 0) {
            build_jacobian_warp(w_J, w_T_world, s_twists, s_act_idx,
                                s_mimic_mul, s_mimic_act_idx, s_ancestor_mask,
                                n_joints, n_act, target_jnt, lane);
        }

        // GN step (warp-parallel).
        gn_step_warp(w_J, w_r, w_cfg, w_col_scale, w_A, w_rhs,
                     s_lower, s_upper, s_fixed_mask, W, n_act, lam, lane);
    }

    // ─── Phase 2 init: reset to best_cfg, compute FK + J there ────────────
    for (int a = lane; a < n_act; a += 32) w_cfg[a] = w_best_cfg[a];
    __syncwarp();

    if (lane == 0) {
        fk_single(w_cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                  s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                  w_T_world, n_joints, n_act);
    }
    __syncwarp();

    if (lane == 0) {
        float target_T2[7] = {
            s_target_quat[0], s_target_quat[1], s_target_quat[2], s_target_quat[3],
            target_p[0], target_p[1], target_p[2],
        };
        compute_r_from_T_world(w_T_world, target_jnt, target_T2, w_r);
        const float* ee0 = w_T_world + target_jnt * 7 + 4;
        const float  d0  = box_dist_sq(ee0, w_box_min, w_box_max);
        if (d0 < w_scratch[0]) {
            w_scratch[0] = d0;
            for (int a = 0; a < n_act; a++) w_best_cfg[a] = w_cfg[a];
        }
    }
    __syncwarp();

    build_jacobian_warp(w_J, w_T_world, s_twists, s_act_idx,
                        s_mimic_mul, s_mimic_act_idx, s_ancestor_mask,
                        n_joints, n_act, target_jnt, lane);

    // ─── Phase 2: Null-Space Brownian Shuffle ─────────────────────────────
    for (int step = 0; step < n_brownian_steps; step++) {

        // ── Null-space Brownian step ──────────────────────────────────────
        //   Δq = ξ − J_pos^T (J_pos J_pos^T + εI)^{-1} J_pos ξ,  ξ ~ N(0,σ)
        //
        // Lane a (0 ≤ a < n_act) samples ξ[a]; result stored in w_rhs[a].
        {
            if (lane < n_act)
                w_rhs[lane] = s_fixed_mask[lane] ? 0.0f
                                                  : rng_normal(rng_state) * noise_std;
            __syncwarp();

            // v3 = J_pos @ ξ  (3-element; warp-parallel then reduce)
            float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f;
            for (int a = lane; a < n_act; a += 32) {
                const float xi = w_rhs[a];
                lv0 += w_J[0*n_act + a] * xi;
                lv1 += w_J[1*n_act + a] * xi;
                lv2 += w_J[2*n_act + a] * xi;
            }
            for (int off = 16; off > 0; off >>= 1) {
                lv0 += __shfl_down_sync(0xffffffff, lv0, off);
                lv1 += __shfl_down_sync(0xffffffff, lv1, off);
                lv2 += __shfl_down_sync(0xffffffff, lv2, off);
            }

            if (lane == 0) {
                w_scratch[3] = lv0;
                w_scratch[4] = lv1;
                w_scratch[5] = lv2;

                // B3 = J_pos J_pos^T + εI  (3×3, stride 3 in w_A[0..8]).
                for (int i = 0; i < 3; i++) {
                    for (int jj = i; jj < 3; jj++) {
                        float acc = 0.0f;
                        for (int a = 0; a < n_act; a++)
                            acc += w_J[i*n_act + a] * w_J[jj*n_act + a];
                        w_A[i*3 + jj] = w_A[jj*3 + i] = acc;
                    }
                    w_A[i*3 + i] += 1e-4f;
                }
                // Analytical 3×3 solve: w_scratch[3..5] ← B3^{-1} v3.
                solve3x3_sym(w_A, w_scratch + 3);
            }
            __syncwarp();

            // cfg += ξ − J_pos^T w  (parallel over n_act)
            for (int a = lane; a < n_act; a += 32) {
                const float proj = w_J[0*n_act+a] * w_scratch[3]
                                 + w_J[1*n_act+a] * w_scratch[4]
                                 + w_J[2*n_act+a] * w_scratch[5];
                w_cfg[a] = clampf(w_cfg[a] + w_rhs[a] - proj, s_lower[a], s_upper[a]);
            }
            __syncwarp();
        }

        // ── Periodic FK check and correction ─────────────────────────────
        if ((step % fk_check_freq) == 0) {
            if (lane == 0) {
                fk_single(w_cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                          s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                          w_T_world, n_joints, n_act);
            }
            __syncwarp();

            if (lane == 0) {
                const float* T_ee = w_T_world + target_jnt * 7;
                const float  d    = box_dist_sq(T_ee + 4, w_box_min, w_box_max);
                w_scratch[1] = d;
                if (d < w_scratch[0]) {
                    w_scratch[0] = d;
                    for (int a = 0; a < n_act; a++) w_best_cfg[a] = w_cfg[a];
                }
                // Nearest point on box boundary for corrective target.
                for (int k = 0; k < 3; k++)
                    w_scratch[6+k] = clampf(T_ee[4+k], w_box_min[k], w_box_max[k]);
            }
            __syncwarp();

            // Corrective GN when EE has drifted outside the box.
            // All 32 lanes read w_scratch[1] → no intra-warp divergence.
            if (w_scratch[1] > 1e-8f) {
                const float corr_T[7] = {
                    s_target_quat[0], s_target_quat[1], s_target_quat[2], s_target_quat[3],
                    w_scratch[6], w_scratch[7], w_scratch[8],
                };

                // 2 corrective GN steps; each call refreshes w_J as a
                // side-effect, keeping the null-space approximation current.
                for (int corr = 0; corr < 2; corr++) {
                    if (lane == 0) {
                        fk_single(w_cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                                  s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                                  w_T_world, n_joints, n_act);
                    }
                    __syncwarp();

                    if (lane == 0)
                        compute_r_from_T_world(w_T_world, target_jnt, corr_T, w_r);
                    __syncwarp();

                    build_jacobian_warp(w_J, w_T_world, s_twists, s_act_idx,
                                        s_mimic_mul, s_mimic_act_idx, s_ancestor_mask,
                                        n_joints, n_act, target_jnt, lane);

                    gn_step_warp(w_J, w_r, w_cfg, w_col_scale, w_A, w_rhs,
                                 s_lower, s_upper, s_fixed_mask, W, n_act, lam, lane);
                }

                // FK on corrected cfg; update best.
                if (lane == 0) {
                    fk_single(w_cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                              s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                              w_T_world, n_joints, n_act);
                }
                __syncwarp();

                if (lane == 0) {
                    const float* T_ee_c = w_T_world + target_jnt * 7;
                    const float  d_c    = box_dist_sq(T_ee_c + 4, w_box_min, w_box_max);
                    if (d_c < w_scratch[0]) {
                        w_scratch[0] = d_c;
                        for (int a = 0; a < n_act; a++) w_best_cfg[a] = w_cfg[a];
                    }
                }
                __syncwarp();
                // w_J is now the Jacobian from the last corrective step,
                // approximately current → valid for subsequent null-space steps.
            }
        }
    }

    // ── Final FK on best_cfg for output ────────────────────────────────────
    if (lane == 0) {
        fk_single(w_best_cfg, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                  s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                  w_T_world, n_joints, n_act);
    }
    __syncwarp();

    const float* T_ee_out = w_T_world + target_jnt * 7;

    // Write outputs (cfg parallel; scalars and 3-vectors on lane 0).
    for (int a = lane; a < n_act; a += 32)
        out_cfg[warp_global_id * n_act + a] = w_best_cfg[a];

    if (lane == 0) {
        out_err[warp_global_id]               = w_scratch[0];
        out_ee_points[warp_global_id * 3 + 0] = T_ee_out[4];
        out_ee_points[warp_global_id * 3 + 1] = T_ee_out[5];
        out_ee_points[warp_global_id * 3 + 2] = T_ee_out[6];
        out_target_points[warp_global_id * 3 + 0] = target_p[0];
        out_target_points[warp_global_id * 3 + 1] = target_p[1];
        out_target_points[warp_global_id * 3 + 2] = target_p[2];
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error BrownianMotionIkCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> seeds,
    ffi::Buffer<ffi::DataType::F32> init_points,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::S32> ancestor_mask,
    ffi::Buffer<ffi::DataType::F32> target_quat,
    ffi::Buffer<ffi::DataType::F32> box_mins,   // (n_problems, 3) — per-problem
    ffi::Buffer<ffi::DataType::F32> box_maxs,   // (n_problems, 3) — per-problem
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    ffi::Buffer<ffi::DataType::S32> rng_seed,
    int64_t target_jnt,
    int64_t max_iter,
    float   pos_weight,
    float   ori_weight,
    float   lambda_init,
    float   eps_pos,
    float   noise_std,
    int64_t n_brownian_steps,
    int64_t fk_check_freq,
    int64_t threads_per_block,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_cfg,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_ee_points,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_target_points)
{
    const int n_problems = static_cast<int>(seeds.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    const int n_act      = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);

    if (n_act > MAX_ACT || n_joints > MAX_JOINTS) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "BrownianMotionIkCuda: compile-time limits exceeded (MAX_ACT/MAX_JOINTS)."
        );
    }

    // threads_per_block must be a multiple of 32 (warp size) and ≤ 1024.
    const int tpb = static_cast<int>(threads_per_block);
    if (tpb < 32 || tpb > 1024 || tpb % 32 != 0) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "BrownianMotionIkCuda: threads_per_block must be a multiple of 32 in [32, 1024]."
        );
    }

    // Each 32-thread warp handles one (problem, seed) pair.
    const int warps_per_block = tpb / 32;
    const int total_warps     = n_problems * n_seeds;
    const int blocks          = (total_warps + warps_per_block - 1) / warps_per_block;

    // Dynamic shared memory: per-warp workspace.
    const size_t smem_bytes = static_cast<size_t>(warps_per_block)
                            * WSMEM_PER_WARP_F * sizeof(float);
    const size_t total_smem_bytes = smem_bytes + STATIC_SMEM_BYTES;

    int device = -1;
    cudaError_t cuda_err = cudaGetDevice(&device);
    if (cuda_err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(cuda_err));
    }

    int max_smem_per_block = 0;
    cuda_err = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (cuda_err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(cuda_err));
    }
    if (total_smem_bytes > static_cast<size_t>(max_smem_per_block)) {
        const size_t dyn_budget =
            (max_smem_per_block > static_cast<int>(STATIC_SMEM_BYTES))
            ? static_cast<size_t>(max_smem_per_block) - STATIC_SMEM_BYTES
            : 0;
        const int max_warps_by_smem =
            static_cast<int>(dyn_budget / (WSMEM_PER_WARP_F * sizeof(float)));
        const int max_tpb_by_smem = 32 * max_warps_by_smem;

        char msg[320];
        std::snprintf(
            msg,
            sizeof(msg),
            "BrownianMotionIkCuda: threads_per_block=%d exceeds shared-memory budget "
            "(requested %zu B total, device limit %d B). "
            "For this build, max threads_per_block by shared memory is %d.",
            tpb,
            total_smem_bytes,
            max_smem_per_block,
            max_tpb_by_smem);
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, msg);
    }

    // Clear any pre-existing CUDA error on this thread before our launch.
    (void)cudaGetLastError();

    brownian_motion_ik_kernel<<<dim3(blocks), tpb, smem_bytes, stream>>>(
        seeds.typed_data(),
        init_points.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        ancestor_mask.typed_data(),
        target_quat.typed_data(),
        box_mins.typed_data(),
        box_maxs.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        rng_seed.typed_data(),
        out_cfg->typed_data(),
        out_err->typed_data(),
        out_ee_points->typed_data(),
        out_target_points->typed_data(),
        n_problems,
        n_seeds,
        n_joints,
        n_act,
        static_cast<int>(target_jnt),
        static_cast<int>(max_iter),
        pos_weight,
        ori_weight,
        lambda_init,
        eps_pos,
        noise_std,
        static_cast<int>(n_brownian_steps),
        static_cast<int>(fk_check_freq));

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BrownianMotionIkCudaFfi,
    BrownianMotionIkCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // seeds
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // init_points
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // ancestor_mask
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // target_quat
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // box_mins  (n_problems, 3)
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // box_maxs  (n_problems, 3)
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // fixed_mask
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // rng_seed
        .Attr<int64_t>("target_jnt")
        .Attr<int64_t>("max_iter")
        .Attr<float>("pos_weight")
        .Attr<float>("ori_weight")
        .Attr<float>("lambda_init")
        .Attr<float>("eps_pos")
        .Attr<float>("noise_std")
        .Attr<int64_t>("n_brownian_steps")
        .Attr<int64_t>("fk_check_freq")
        .Attr<int64_t>("threads_per_block")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_cfg
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_err
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_ee_points
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out_target_points
);
