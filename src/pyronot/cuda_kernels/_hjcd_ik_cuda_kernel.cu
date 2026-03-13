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
 * FK is performed by calling fk_single() from _fk_cuda_helpers.cuh so that
 * the FK logic is not duplicated.
 *
 * Numerical stability:
 *   - FK and Jacobian in float32.
 *   - Normal-equation matrix and Cholesky solve in float64.
 *   - All kernel launches are associated with the caller's CUDA stream so
 *     there are no implicit device synchronisations.
 *
 * Jacobian convention:
 *   The geometric (world-frame) Jacobian is used together with a world-frame
 *   residual:
 *     r_pos = p_target - p_ee                         (3-vector)
 *     r_ori = rotvec( q_target * conj(q_ee) )         (3-vector)
 *   For revolute joint j:  J_lin = z_j x (p_ee - p_j),  J_ang = z_j
 *   For prismatic joint j: J_lin = z_j,                  J_ang = 0
 *   where z_j = R(T_world[j]) * body_axis[j].
 *   Non-ancestor joints are zeroed out via the ancestor_mask input.
 *
 * Build with:  bash src/pyronot/cuda_kernels/build_hjcd_ik_cuda.sh
 */

#include "_fk_cuda_helpers.cuh"

#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits
// ---------------------------------------------------------------------------

// Maximum joint / actuated-joint counts.  Increase if needed for large robots.
// Keep these as small as possible: CUDA pre-allocates local memory for
// MAX_RESIDENT_THREADS × per-thread stack, which scales as MAX_ACT^2 (from
// the double A_s[MAX_ACT*MAX_ACT] normal-equation matrix in ik_lm_kernel).
// With MAX_ACT=64 this exceeds 4 GB on a 68-SM GPU; with 16 it is ~730 MB.
#define MAX_JOINTS 32
#define MAX_ACT    16

// ---------------------------------------------------------------------------
// Math helpers (IK-specific)
// ---------------------------------------------------------------------------

/** Cross product: out = a x b. */
__device__ __forceinline__
void cross3(const float* __restrict__ a,
            const float* __restrict__ b,
            float* __restrict__ out)
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/** Dot product of two 3-vectors. */
__device__ __forceinline__
float dot3(const float* __restrict__ a, const float* __restrict__ b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/** L2 norm of a 3-vector. */
__device__ __forceinline__
float norm3(const float* __restrict__ v)
{
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/** Clamp x to [lo, hi]. */
__device__ __forceinline__
float clampf(float x, float lo, float hi)
{
    return fmaxf(lo, fminf(hi, x));
}

// ---------------------------------------------------------------------------
// Cholesky solver (sequential, float64, in-place)
// ---------------------------------------------------------------------------

/**
 * Solve  A x = b  via Cholesky decomposition (LL^T = A).
 *
 * A is n×n symmetric positive semi-definite, stored row-major.
 * On return b holds the solution x.  A is overwritten with L.
 *
 * Returns true on success; false if A is not positive-definite.
 * On failure, x is set to zero.
 */
__device__ bool chol_solve(double* __restrict__ A,
                           double* __restrict__ b,
                           int n)
{
    // Factorization: L L^T = A
    for (int k = 0; k < n; k++) {
        double s = A[k*n + k];
        for (int p = 0; p < k; p++) { double lkp = A[k*n+p]; s -= lkp*lkp; }
        if (s <= 0.0) {
            for (int i = 0; i < n; i++) b[i] = 0.0;
            return false;
        }
        double lkk = sqrt(s);
        A[k*n + k] = lkk;
        for (int i = k+1; i < n; i++) {
            double t = A[i*n + k];
            for (int p = 0; p < k; p++) t -= A[i*n+p] * A[k*n+p];
            A[i*n + k] = t / lkk;
        }
        // Zero upper triangle (not required for the solve, but keeps A tidy).
        for (int j = k+1; j < n; j++) A[k*n + j] = 0.0;
    }

    // Forward substitution: L y = b
    double y[MAX_ACT];
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int p = 0; p < i; p++) s -= A[i*n+p] * y[p];
        y[i] = s / A[i*n + i];
    }

    // Backward substitution: L^T x = y
    for (int i = n-1; i >= 0; i--) {
        double s = y[i];
        for (int p = i+1; p < n; p++) s -= A[p*n + i] * b[p];
        b[i] = s / A[i*n + i];
    }
    return true;
}

// ---------------------------------------------------------------------------
// IK residual and geometric Jacobian
// ---------------------------------------------------------------------------

/**
 * Compute the IK residual r (6-vector) and the world-frame geometric
 * Jacobian J (6 × n_act) for a given joint configuration.
 *
 * FK is performed internally via fk_single().
 *
 * @param cfg             (n_act,)       current actuated configuration
 * @param T_world         (n_joints, 7)  scratch space; filled with FK output
 * @param target_T        [w,x,y,z,tx,ty,tz]  target end-effector pose
 * @param target_jnt      joint index used as end-effector
 * @param ancestor_mask   (n_joints,) int32; 1 if joint contributes to EE
 * @param r               (6,) output residual  [r_pos(3), r_ori(3)]
 * @param J               (6, n_act) output Jacobian, row-major
 */
__device__ void compute_residual_and_jacobian(
    const float* __restrict__ cfg,
    float*       __restrict__ T_world,       // scratch, (n_joints, 7)
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ ancestor_mask,
    const float* __restrict__ target_T,
    int target_jnt,
    int n_joints, int n_act,
    float* __restrict__ r,
    float* __restrict__ J)
{
    // ---- FK ----------------------------------------------------------------
    fk_single(cfg, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv,
              T_world, n_joints, n_act);

    // ---- Residual ----------------------------------------------------------
    const float* T_ee = T_world + target_jnt * 7;
    const float p_ee[3] = { T_ee[4], T_ee[5], T_ee[6] };
    const float q_ee[4] = { T_ee[0], T_ee[1], T_ee[2], T_ee[3] };

    const float p_tgt[3] = { target_T[4], target_T[5], target_T[6] };
    const float q_tgt[4] = { target_T[0], target_T[1], target_T[2], target_T[3] };

    // Position error (world frame).
    // Convention: r = p_ee - p_tgt so that ∂r/∂q_i = +J_lin_i (consistent
    // with the world-frame geometric Jacobian stored in J below).
    r[0] = p_ee[0] - p_tgt[0];
    r[1] = p_ee[1] - p_tgt[1];
    r[2] = p_ee[2] - p_tgt[2];

    // Orientation error as rotation vector.
    // q_err = q_ee * conj(q_tgt)  so that ∂r_ori/∂q_i ≈ +J_ang_i near
    // convergence, consistent with the world-frame angular Jacobian below.
    const float q_tgt_inv[4] = { q_tgt[0], -q_tgt[1], -q_tgt[2], -q_tgt[3] };
    float q_err[4];
    quat_mul(q_ee, q_tgt_inv, q_err);
    // Ensure shortest path.
    if (q_err[0] < 0.0f) {
        q_err[0] = -q_err[0]; q_err[1] = -q_err[1];
        q_err[2] = -q_err[2]; q_err[3] = -q_err[3];
    }
    // Convert to rotation vector: 2*atan2(||vec||, w) * vec/||vec||.
    const float sin_half = sqrtf(q_err[1]*q_err[1] + q_err[2]*q_err[2] + q_err[3]*q_err[3]);
    float theta;
    if (sin_half > 1e-6f) {
        theta = 2.0f * atan2f(sin_half, q_err[0]);
        const float inv_sin = theta / sin_half;
        r[3] = q_err[1] * inv_sin;
        r[4] = q_err[2] * inv_sin;
        r[5] = q_err[3] * inv_sin;
    } else {
        // Small angle: rotvec ≈ 2 * q_err.xyz
        r[3] = 2.0f * q_err[1];
        r[4] = 2.0f * q_err[2];
        r[5] = 2.0f * q_err[3];
    }

    // ---- Geometric Jacobian ------------------------------------------------
    // Zero all columns first.
    for (int i = 0; i < 6 * n_act; i++) J[i] = 0.0f;

    const float arm[3] = { p_ee[0], p_ee[1], p_ee[2] };

    for (int j = 0; j < n_joints; j++) {
        if (!ancestor_mask[j]) continue;

        // Determine which actuated joint(s) this joint feeds into.
        int a1 = act_idx[j];       // direct actuated index (-1 if fixed)
        int a2 = mimic_act_idx[j]; // mimicked actuated index (-1 if not mimic)
        if (a1 < 0 && a2 < 0) continue; // fixed joint, no contribution

        // Body-frame twist axis for this joint.
        const float* tw = twists + j * 6;
        const float ang_sq = tw[3]*tw[3] + tw[4]*tw[4] + tw[5]*tw[5];
        const float lin_sq = tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2];

        const float* T_j  = T_world + j * 7;
        const float* q_j  = T_j;        // quaternion [w,x,y,z]
        const float* p_j  = T_j + 4;    // translation [tx,ty,tz]

        float jg_lin[3], jg_ang[3];

        if (ang_sq > 1e-6f) {
            // Revolute: body axis from angular part of twist.
            const float inv_ang = 1.0f / sqrtf(ang_sq);
            const float body_ax[3] = { tw[3]*inv_ang, tw[4]*inv_ang, tw[5]*inv_ang };
            // World-frame axis.
            float z_j[3];
            quat_rotate(q_j, body_ax, z_j);
            // Jacobian: J_lin = z x (p_ee - p_j),  J_ang = z.
            const float arm_j[3] = { arm[0]-p_j[0], arm[1]-p_j[1], arm[2]-p_j[2] };
            cross3(z_j, arm_j, jg_lin);
            jg_ang[0] = z_j[0]; jg_ang[1] = z_j[1]; jg_ang[2] = z_j[2];
        } else if (lin_sq > 1e-6f) {
            // Prismatic: body axis from linear part of twist.
            const float inv_lin = 1.0f / sqrtf(lin_sq);
            const float body_ax[3] = { tw[0]*inv_lin, tw[1]*inv_lin, tw[2]*inv_lin };
            float z_j[3];
            quat_rotate(q_j, body_ax, z_j);
            jg_lin[0] = z_j[0]; jg_lin[1] = z_j[1]; jg_lin[2] = z_j[2];
            jg_ang[0] = 0.0f;   jg_ang[1] = 0.0f;   jg_ang[2] = 0.0f;
        } else {
            // Fixed: zero contribution.
            continue;
        }

        // Accumulate into actuated Jacobian columns (with mimic scaling).
        // For a direct joint, mimic_mul == 1.0; for a mimic, it's the multiplier.
        if (a1 >= 0) {
            const float s = mimic_mul[j]; // 1.0 for non-mimic direct joints
            J[0*n_act + a1] += s * jg_lin[0];
            J[1*n_act + a1] += s * jg_lin[1];
            J[2*n_act + a1] += s * jg_lin[2];
            J[3*n_act + a1] += s * jg_ang[0];
            J[4*n_act + a1] += s * jg_ang[1];
            J[5*n_act + a1] += s * jg_ang[2];
        }
        if (a2 >= 0) {
            const float s = mimic_mul[j];
            J[0*n_act + a2] += s * jg_lin[0];
            J[1*n_act + a2] += s * jg_lin[1];
            J[2*n_act + a2] += s * jg_lin[2];
            J[3*n_act + a2] += s * jg_ang[0];
            J[4*n_act + a2] += s * jg_ang[1];
            J[5*n_act + a2] += s * jg_ang[2];
        }
    }
}

// ---------------------------------------------------------------------------
// Residual-only evaluation (no Jacobian) — used by line search
// ---------------------------------------------------------------------------

/**
 * Compute only the IK residual r (6-vector) without the Jacobian.
 * Much cheaper than compute_residual_and_jacobian for cost-only evaluation
 * (e.g., line search candidates).
 */
__device__ void compute_residual_only(
    const float* __restrict__ cfg,
    float*       __restrict__ T_world,       // scratch, (n_joints, 7)
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const float* __restrict__ target_T,
    int target_jnt,
    int n_joints, int n_act,
    float* __restrict__ r)
{
    // ---- FK ----------------------------------------------------------------
    fk_single(cfg, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv,
              T_world, n_joints, n_act);

    // ---- Residual ----------------------------------------------------------
    const float* T_ee = T_world + target_jnt * 7;
    const float p_ee[3] = { T_ee[4], T_ee[5], T_ee[6] };
    const float q_ee[4] = { T_ee[0], T_ee[1], T_ee[2], T_ee[3] };

    const float p_tgt[3] = { target_T[4], target_T[5], target_T[6] };
    const float q_tgt[4] = { target_T[0], target_T[1], target_T[2], target_T[3] };

    r[0] = p_ee[0] - p_tgt[0];
    r[1] = p_ee[1] - p_tgt[1];
    r[2] = p_ee[2] - p_tgt[2];

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
 * At each step:
 *   1. FK + residual + Jacobian.
 *   2. Adaptive row weighting.
 *   3. For each actuated joint a: compute predicted improvement
 *        Δa = (Jw[:,a]^T fw)^2 / (||Jw[:,a]||^2 + eps).
 *   4. Apply optimal step for the joint with maximum Δ (fixed joints skipped).
 *   5. Clip to joint limits.
 *
 * @param seeds          (n_seeds, n_act)  initial configurations
 * @param twists         (n_joints, 6)
 * @param parent_tf      (n_joints, 7)
 * @param parent_idx     (n_joints,)
 * @param act_idx        (n_joints,)
 * @param mimic_mul      (n_joints,)
 * @param mimic_off      (n_joints,)
 * @param mimic_act_idx  (n_joints,)
 * @param topo_inv       (n_joints,)
 * @param ancestor_mask  (n_joints,) int32
 * @param target_T       (7,)  target pose [w,x,y,z,tx,ty,tz]
 * @param lower          (n_act,) lower limits
 * @param upper          (n_act,) upper limits
 * @param fixed_mask     (n_act,) int32; 1 = fixed, 0 = free
 * @param out            (n_seeds, n_act)  output configurations
 * @param n_seeds, n_joints, n_act, target_jnt, k_max
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
    const int*   __restrict__ ancestor_mask,
    const float* __restrict__ target_T,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    int n_problems, int n_seeds, int n_joints, int n_act, int target_jnt, int k_max)
{
    // ── Shared memory: robot parameters loaded once per block ───────────────
    // All threads cooperate to copy constant robot data from global memory
    // into shared (L1-backed) memory (~2.8 KB/block).  Every subsequent FK
    // and Jacobian call then reads from shared memory instead of global DRAM.
    // The early-exit guard is placed AFTER __syncthreads so out-of-range
    // threads still participate in the cooperative load.
    __shared__ float s_twists       [MAX_JOINTS * 6];
    __shared__ float s_parent_tf    [MAX_JOINTS * 7];
    __shared__ int   s_parent_idx   [MAX_JOINTS];
    __shared__ int   s_act_idx      [MAX_JOINTS];
    __shared__ float s_mimic_mul    [MAX_JOINTS];
    __shared__ float s_mimic_off    [MAX_JOINTS];
    __shared__ int   s_mimic_act_idx[MAX_JOINTS];
    __shared__ int   s_topo_inv     [MAX_JOINTS];
    __shared__ int   s_ancestor_mask[MAX_JOINTS];
    __shared__ float s_target_T[7];
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
        s_ancestor_mask[i] = ancestor_mask[i];
    }
    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[i];
        s_upper[i]      = upper[i];
        s_fixed_mask[i] = fixed_mask[i];
    }
    const int p = blockIdx.y;
    if (threadIdx.x < 7) s_target_T[threadIdx.x] = target_T[p * 7 + threadIdx.x];
    __syncthreads();  // All robot data visible before any thread begins FK.

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int gs = p * n_seeds + s;

    // Local configuration (thread-private).
    float cfg[MAX_ACT];
    for (int a = 0; a < n_act; a++) cfg[a] = seeds[gs * n_act + a];

    // Scratch for FK world transforms and Jacobian.
    float T_world[MAX_JOINTS * 7];
    float r[6], J[6 * MAX_ACT];

    for (int iter = 0; iter < k_max; iter++) {
        compute_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_ancestor_mask, s_target_T, target_jnt,
            n_joints, n_act, r, J);

        // Adaptive weighting with orientation gating.
        // Orientation contribution is disabled until position error < 1 mm,
        // matching the reference HJCD coarse phase (s_allow_ori = pos < 1mm).
        float w[6];
        adaptive_weights(r, w);
        {
            const float pe = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
            if (pe >= 1e-3f) { w[3] = 0.0f; w[4] = 0.0f; w[5] = 0.0f; }
        }

        // fw = r * w,  Jw = J * w (row-wise).
        float fw[6];
        for (int k = 0; k < 6; k++) fw[k] = r[k] * w[k];
        float Jw[6 * MAX_ACT];
        for (int k = 0; k < 6; k++)
            for (int a = 0; a < n_act; a++)
                Jw[k * n_act + a] = J[k * n_act + a] * w[k];

        // Per-joint: JwTfw[a] = Jw[:,a]^T fw,  Jw_normsq[a] = ||Jw[:,a]||^2.
        float JwTfw[MAX_ACT], Jw_normsq[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float s_dot = 0.0f, s_sq = 0.0f;
            for (int k = 0; k < 6; k++) {
                const float jwa = Jw[k * n_act + a];
                s_dot += jwa * fw[k];
                s_sq  += jwa * jwa;
            }
            JwTfw[a]    = s_dot;
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

    // Compute final error for scoring (avoids Python-side FK + SE3 log).
    compute_residual_only(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_target_T, target_jnt, n_joints, n_act, r);
    float final_err = 0.0f;
    for (int k = 0; k < 6; k++) final_err += r[k] * r[k];
    out_err[gs] = final_err;

    // Write output.
    for (int a = 0; a < n_act; a++) out[gs * n_act + a] = cfg[a];
}

// ---------------------------------------------------------------------------
// LM refinement kernel (Phase 2)
// ---------------------------------------------------------------------------

/**
 * One CUDA thread per refinement seed.  Runs max_iter LM iterations with:
 *   - Adaptive pos/ori row weighting.
 *   - Jacobi column scaling.
 *   - Soft joint-limit prior.
 *   - Line search over {1, 0.5, 0.25, 0.125} step multipliers.
 *   - Stall detection with random kicks (noise pre-generated in Python).
 *   - Best-config tracking (kicks never degrade the returned result).
 *
 * @param seeds         (n_seeds, n_act)
 * @param noise         (n_seeds, max_iter, n_act)  kick noise
 * @param [robot params ...]
 * @param target_T      (7,)  target pose
 * @param lower/upper   (n_act,) limits
 * @param fixed_mask    (n_act,) int32
 * @param out           (n_seeds, n_act)  best configs
 * @param lambda_init, limit_prior_weight, kick_scale, eps_pos, eps_ori: floats
 * @param stall_patience, max_iter: ints
 */
__global__
void hjcd_ik_lm_kernel(
    const float* __restrict__ seeds,
    const float* __restrict__ noise,         // (n_seeds, max_iter, n_act)
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    const int*   __restrict__ ancestor_mask,
    const float* __restrict__ target_T,
    const float* __restrict__ lower,
    const float* __restrict__ upper,
    const int*   __restrict__ fixed_mask,
    float*       __restrict__ out,
    float*       __restrict__ out_err,
    int*         __restrict__ stop_flag,
    int n_problems, int n_seeds, int n_joints, int n_act, int target_jnt, int max_iter,
    float lambda_init, float limit_prior_weight, float kick_scale,
    float eps_pos, float eps_ori, int stall_patience)
{
    // ── Shared memory: robot parameters loaded once per block ───────────────
    // All threads cooperate to copy constant robot data from global memory
    // into shared (L1-backed) memory (~2.8 KB/block).  Every subsequent FK
    // and Jacobian call then reads from shared memory instead of global DRAM.
    // The early-exit guard is placed AFTER __syncthreads so out-of-range
    // threads still participate in the cooperative load.
    __shared__ float s_twists       [MAX_JOINTS * 6];
    __shared__ float s_parent_tf    [MAX_JOINTS * 7];
    __shared__ int   s_parent_idx   [MAX_JOINTS];
    __shared__ int   s_act_idx      [MAX_JOINTS];
    __shared__ float s_mimic_mul    [MAX_JOINTS];
    __shared__ float s_mimic_off    [MAX_JOINTS];
    __shared__ int   s_mimic_act_idx[MAX_JOINTS];
    __shared__ int   s_topo_inv     [MAX_JOINTS];
    __shared__ int   s_ancestor_mask[MAX_JOINTS];
    __shared__ float s_target_T[7];
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
        s_ancestor_mask[i] = ancestor_mask[i];
    }
    for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
        s_lower[i]      = lower[i];
        s_upper[i]      = upper[i];
        s_fixed_mask[i] = fixed_mask[i];
    }
    const int p = blockIdx.y;
    if (threadIdx.x < 7) s_target_T[threadIdx.x] = target_T[p * 7 + threadIdx.x];
    __syncthreads();  // All robot data visible before any thread begins FK.

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int gs = p * n_seeds + s;

    // Thread-private state.
    float cfg[MAX_ACT], best_cfg[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[6], J[6 * MAX_ACT];

    // Load initial config.
    for (int a = 0; a < n_act; a++) cfg[a] = seeds[gs * n_act + a];
    for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];

    // Joint-limit mid / half-range for prior.
    float mid[MAX_ACT], half_range[MAX_ACT];
    for (int a = 0; a < n_act; a++) {
        mid[a]        = (s_lower[a] + s_upper[a]) * 0.5f;
        half_range[a] = (s_upper[a] - s_lower[a]) * 0.5f + 1e-8f;
    }

    // Compute initial squared error.
    compute_residual_and_jacobian(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_ancestor_mask, s_target_T, target_jnt,
        n_joints, n_act, r, J);
    float best_err = 0.0f;
    for (int k = 0; k < 6; k++) best_err += r[k] * r[k];

    float lam   = lambda_init;
    int   stall = 0;
    bool  done  = false;

    for (int iter = 0; iter < max_iter; iter++) {
        if (done) break;
        if (stop_flag[p]) break;  // Another seed in this problem converged

        // ── Jacobian + residual ──────────────────────────────────────────
        compute_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_ancestor_mask, s_target_T, target_jnt,
            n_joints, n_act, r, J);

        float curr_err = 0.0f;
        for (int k = 0; k < 6; k++) curr_err += r[k] * r[k];

        // Early exit check on best solution.
        float r_pos = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        float r_ori = sqrtf(r[3]*r[3] + r[4]*r[4] + r[5]*r[5]);
        if (r_pos < eps_pos && r_ori < eps_ori) {
            done = true;
            atomicExch(stop_flag + p, 1);
            __threadfence();  // Ensure other blocks see the convergence flag
            break;
        }

        // ── Row weighting ────────────────────────────────────────────────
        float w[6];
        adaptive_weights(r, w);
        float fw[6];
        for (int k = 0; k < 6; k++) fw[k] = r[k] * w[k];
        float Jw[6 * MAX_ACT];
        for (int k = 0; k < 6; k++)
            for (int a = 0; a < n_act; a++)
                Jw[k * n_act + a] = J[k * n_act + a] * w[k];

        // ── Jacobi column scaling ─────────────────────────────────────────
        float col_scale[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float sq = 0.0f;
            for (int k = 0; k < 6; k++) { float v = Jw[k*n_act+a]; sq += v*v; }
            col_scale[a] = sqrtf(sq) + 1e-8f;
        }
        // Js = Jw / col_scale (column-normalised).
        float Js[6 * MAX_ACT];
        for (int k = 0; k < 6; k++)
            for (int a = 0; a < n_act; a++)
                Js[k * n_act + a] = Jw[k * n_act + a] / col_scale[a];

        // ── Normal equations (float64 for numerical stability) ───────────
        // A_s = Js^T Js + diag(lam + D_prior_s)
        // rhs_s = -(Js^T fw + g_prior_s)
        double A_s[MAX_ACT * MAX_ACT];
        double rhs_s[MAX_ACT];

        for (int i = 0; i < n_act; i++) {
            for (int j = 0; j < n_act; j++) {
                double acc = 0.0;
                for (int k = 0; k < 6; k++)
                    acc += (double)Js[k*n_act+i] * (double)Js[k*n_act+j];
                A_s[i*n_act + j] = acc;
            }
            double rb = 0.0;
            for (int k = 0; k < 6; k++)
                rb += (double)Js[k*n_act+i] * (double)fw[k];
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

        // Trust-region step clipping (radius scales with distance from solution).
        {
            const float p_r = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
            const float o_r = sqrtf(r[3]*r[3] + r[4]*r[4] + r[5]*r[5]);
            float R;
            if      (p_r > 1e-2f || o_r > 0.6f)  R = 0.38f;
            else if (p_r > 1e-3f || o_r > 0.25f) R = 0.22f;
            else if (p_r > 2e-4f || o_r > 0.08f) R = 0.12f;
            else                                   R = 0.05f;
            float dnorm = 0.0f;
            for (int a = 0; a < n_act; a++) dnorm += delta[a]*delta[a];
            dnorm = sqrtf(dnorm);
            if (dnorm > R) {
                const float scale = R / (dnorm + 1e-18f);
                for (int a = 0; a < n_act; a++) delta[a] *= scale;
            }
        }

        // ── Line search: 5 candidates with early exit ─────────────────────
        // Uses residual-only evaluation (no Jacobian) and breaks on first
        // step with sufficient descent, matching HJCD-IK's backtracking.
        const float alphas[5] = { 1.0f, 0.5f, 0.25f, 0.1f, 0.025f };
        float best_alpha_err  = 1e30f;
        int   best_alpha_idx  = 0;
        float r_trial[6];

        for (int ai = 0; ai < 5; ai++) {
            float cfg_trial[MAX_ACT];
            for (int a = 0; a < n_act; a++)
                cfg_trial[a] = clampf(cfg[a] + alphas[ai] * delta[a],
                                      s_lower[a], s_upper[a]);
            // Residual-only evaluation — skips expensive Jacobian assembly.
            // Reuses T_world scratch (recomputed at start of next iteration).
            compute_residual_only(
                cfg_trial, T_world,
                s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                s_target_T, target_jnt, n_joints, n_act, r_trial);
            float err_trial = 0.0f;
            for (int k = 0; k < 6; k++) err_trial += r_trial[k] * r_trial[k];
            if (err_trial < best_alpha_err) {
                best_alpha_err = err_trial;
                best_alpha_idx = ai;
                // Early exit: accept first step with sufficient descent.
                if (err_trial < curr_err * (1.0f - 1e-4f)) break;
            }
        }

        // Compute the winning trial config once; reused for both accept and
        // best-tracking to avoid the double-step bug (if cfg were updated
        // first, recomputing cfg + alpha*delta would apply the step twice).
        float trial_cfg[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            trial_cfg[a] = clampf(cfg[a] + alphas[best_alpha_idx] * delta[a],
                                  s_lower[a], s_upper[a]);

        // Accept / reject.
        // Drift guard: require at least 1e-4 relative improvement so that
        // pure floating-point noise cannot cause false acceptances that
        // gradually perturb the trajectory.
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

/**
 * Common robot-model argument list shared by both coarse and LM handlers.
 * Seeds and extra inputs are listed before calling the kernel.
 */
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
    ffi::Buffer<ffi::DataType::S32> ancestor_mask,
    ffi::Buffer<ffi::DataType::F32> target_T,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t target_jnt,
    int64_t k_max,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    const int n_problems = static_cast<int>(target_T.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    const int n_act      = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);

    // Avoid over-launching threads: each thread has a sizable local stack.
    // Launch only as many threads as needed (up to a safe cap).
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
        ancestor_mask.typed_data(),
        target_T.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        out->typed_data(),
        out_err->typed_data(),
        n_problems, n_seeds, n_joints, n_act,
        static_cast<int>(target_jnt),
        static_cast<int>(k_max));

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
    ffi::Buffer<ffi::DataType::S32> ancestor_mask,
    ffi::Buffer<ffi::DataType::F32> target_T,
    ffi::Buffer<ffi::DataType::F32> lower,
    ffi::Buffer<ffi::DataType::F32> upper,
    ffi::Buffer<ffi::DataType::S32> fixed_mask,
    int64_t target_jnt,
    int64_t max_iter,
    int64_t stall_patience,
    float   lambda_init,
    float   limit_prior_weight,
    float   kick_scale,
    float   eps_pos,
    float   eps_ori,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> stop_flag)
{
    const int n_problems = static_cast<int>(target_T.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    const int n_act      = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);

    // LM uses much more per-thread local memory than coarse CD.
    // Keep the block size conservative to prevent launch-time OOM.
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
        ancestor_mask.typed_data(),
        target_T.typed_data(),
        lower.typed_data(),
        upper.typed_data(),
        fixed_mask.typed_data(),
        out->typed_data(),
        out_err->typed_data(),
        stop_flag->typed_data(),
        n_problems, n_seeds, n_joints, n_act,
        static_cast<int>(target_jnt),
        static_cast<int>(max_iter),
        lambda_init, limit_prior_weight, kick_scale,
        eps_pos, eps_ori,
        static_cast<int>(stall_patience));

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
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // ancestor_mask
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_T
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fixed_mask
        .Attr<int64_t>("target_jnt")
        .Attr<int64_t>("k_max")
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
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // ancestor_mask
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // target_T
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fixed_mask
        .Attr<int64_t>("target_jnt")
        .Attr<int64_t>("max_iter")
        .Attr<int64_t>("stall_patience")
        .Attr<float>("lambda_init")
        .Attr<float>("limit_prior_weight")
        .Attr<float>("kick_scale")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // out
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // out_err
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // stop_flag
