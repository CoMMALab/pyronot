/**
 * Gauss-Newton Least Squares IK CUDA kernel with XLA FFI binding.
 *
 * Implements multi-seed Levenberg-Marquardt IK directly (no coarse phase):
 *   - One CUDA thread per seed.
 *   - Fixed pos_weight / ori_weight instead of adaptive row-equilibration.
 *   - Jacobi column scaling in the normal equations.
 *   - 5-point line search (early exit on sufficient descent).
 *   - Trust-region step-size schedule.
 *   - All-time best-config tracking.
 *   - No stall kicks, no joint-limit prior.
 *
 * Reuses _fk_cuda_helpers.cuh for SE(3) math and fk_single().
 * Math helpers (cross3, Cholesky, residual+Jacobian) are defined locally
 * to keep this file self-contained.
 *
 * Numerical stability:
 *   - FK and Jacobian in float32.
 *   - Normal-equation matrix and Cholesky solve in float64.
 *
 * Build with:
 *   bash src/pyronot/cuda_kernels/build_ls_ik_cuda.sh
 */

#include "_fk_cuda_helpers.cuh"
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <cstring>

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Compile-time limits (must cover the largest robot you plan to use)
// ---------------------------------------------------------------------------

#define MAX_JOINTS 32
#define MAX_ACT    16

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__
void cross3(const float* __restrict__ a,
            const float* __restrict__ b,
            float* __restrict__ out)
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ __forceinline__
float norm3(const float* __restrict__ v)
{
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

__device__ __forceinline__
float clampf(float x, float lo, float hi)
{
    return fmaxf(lo, fminf(hi, x));
}

// ---------------------------------------------------------------------------
// Cholesky solver (sequential, float64, in-place) — same as HJCD kernel
// ---------------------------------------------------------------------------

__device__ bool chol_solve(double* __restrict__ A,
                           double* __restrict__ b,
                           int n)
{
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
 * Compute IK residual r (6-vector) and world-frame geometric Jacobian
 * J (6 x n_act, row-major) for a given joint configuration.
 *
 * Residual convention (same as HJCD kernel and JAX _ik_residual):
 *   r_pos = p_ee - p_tgt          (so that d r_pos / d q = +J_lin)
 *   r_ori = rotvec(q_ee * conj(q_tgt))
 *
 * Jacobian convention (world-frame geometric):
 *   Revolute  j:  J_lin = z_j x (p_ee - p_j),  J_ang = z_j
 *   Prismatic j:  J_lin = z_j,                  J_ang = 0
 *   z_j = R(T_world[j]) * body_axis[j]
 */
__device__ void compute_residual_and_jacobian(
    const float* __restrict__ cfg,
    float*       __restrict__ T_world,
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
    fk_single(cfg, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv,
              T_world, n_joints, n_act);

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

    for (int i = 0; i < 6 * n_act; i++) J[i] = 0.0f;

    const float arm[3] = { p_ee[0], p_ee[1], p_ee[2] };

    for (int j = 0; j < n_joints; j++) {
        if (!ancestor_mask[j]) continue;
        int a1 = act_idx[j];
        int a2 = mimic_act_idx[j];
        if (a1 < 0 && a2 < 0) continue;

        const float* tw = twists + j * 6;
        const float ang_sq = tw[3]*tw[3] + tw[4]*tw[4] + tw[5]*tw[5];
        const float lin_sq = tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2];

        const float* T_j = T_world + j * 7;
        const float* q_j = T_j;
        const float* p_j = T_j + 4;

        float jg_lin[3], jg_ang[3];

        if (ang_sq > 1e-6f) {
            const float inv_ang   = 1.0f / sqrtf(ang_sq);
            const float body_ax[3] = { tw[3]*inv_ang, tw[4]*inv_ang, tw[5]*inv_ang };
            float z_j[3];
            quat_rotate(q_j, body_ax, z_j);
            const float arm_j[3] = { arm[0]-p_j[0], arm[1]-p_j[1], arm[2]-p_j[2] };
            cross3(z_j, arm_j, jg_lin);
            jg_ang[0] = z_j[0]; jg_ang[1] = z_j[1]; jg_ang[2] = z_j[2];
        } else if (lin_sq > 1e-6f) {
            const float inv_lin   = 1.0f / sqrtf(lin_sq);
            const float body_ax[3] = { tw[0]*inv_lin, tw[1]*inv_lin, tw[2]*inv_lin };
            float z_j[3];
            quat_rotate(q_j, body_ax, z_j);
            jg_lin[0] = z_j[0]; jg_lin[1] = z_j[1]; jg_lin[2] = z_j[2];
            jg_ang[0] = 0.0f;   jg_ang[1] = 0.0f;   jg_ang[2] = 0.0f;
        } else {
            continue;
        }

        if (a1 >= 0) {
            const float s = mimic_mul[j];
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

/**
 * Residual only (no Jacobian), used for the line search.
 */
__device__ void compute_residual_only(
    const float* __restrict__ cfg,
    float*       __restrict__ T_world,
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
    fk_single(cfg, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv,
              T_world, n_joints, n_act);

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

// ---------------------------------------------------------------------------
// LS-IK LM kernel — one thread per seed
// ---------------------------------------------------------------------------

/**
 * Multi-seed Levenberg-Marquardt IK.
 *
 * Each thread independently refines one seed for max_iter iterations.
 *
 * @param seeds        (n_seeds, n_act)   initial configurations
 * @param [robot params ...]              same as HJCD kernel
 * @param target_T     (7,)               target pose [w,x,y,z,tx,ty,tz]
 * @param lower/upper  (n_act,)           joint limits
 * @param fixed_mask   (n_act,) int32     1 = frozen joint
 * @param out          (n_seeds, n_act)   best configurations
 * @param out_err      (n_seeds,)         best weighted squared errors
 * @param pos_weight   scalar             weight on position residual components
 * @param ori_weight   scalar             weight on orientation residual components
 * @param lambda_init  scalar             initial LM damping
 * @param eps_pos      scalar             position convergence threshold [m]
 * @param eps_ori      scalar             orientation convergence threshold [rad]
 * @param max_iter     int                LM iteration budget
 */
__global__
void ls_ik_lm_kernel(
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
    int   n_problems, int n_seeds, int n_joints, int n_act, int target_jnt, int max_iter,
    float pos_weight, float ori_weight, float lambda_init,
    float eps_pos, float eps_ori)
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
    __syncthreads();

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seeds) return;
    const int gs = p * n_seeds + s;

    // ── Thread-private weight vector ─────────────────────────────────────
    // W = [pos_weight x3, ori_weight x3]
    float W[6];
    W[0] = pos_weight; W[1] = pos_weight; W[2] = pos_weight;
    W[3] = ori_weight; W[4] = ori_weight; W[5] = ori_weight;

    // ── Thread-private state ─────────────────────────────────────────────
    float cfg[MAX_ACT], best_cfg[MAX_ACT];
    float T_world[MAX_JOINTS * 7];
    float r[6], J[6 * MAX_ACT];

    for (int a = 0; a < n_act; a++) cfg[a] = seeds[gs * n_act + a];
    for (int a = 0; a < n_act; a++) best_cfg[a] = cfg[a];

    // Initial weighted error.
    compute_residual_and_jacobian(
        cfg, T_world,
        s_twists, s_parent_tf, s_parent_idx, s_act_idx,
        s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
        s_ancestor_mask, s_target_T, target_jnt,
        n_joints, n_act, r, J);
    float best_err = 0.0f;
    for (int k = 0; k < 6; k++) { float rw = r[k] * W[k]; best_err += rw * rw; }

    float lam = lambda_init;

    for (int iter = 0; iter < max_iter; iter++) {

        // ── Residual + Jacobian ─────────────────────────────────────────
        compute_residual_and_jacobian(
            cfg, T_world,
            s_twists, s_parent_tf, s_parent_idx, s_act_idx,
            s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
            s_ancestor_mask, s_target_T, target_jnt,
            n_joints, n_act, r, J);

        // Early exit if converged.
        {
            const float p_r = norm3(r);
            const float o_r = norm3(r + 3);
            if (p_r < eps_pos && o_r < eps_ori) break;
        }

        // Weighted residual and Jacobian.
        float fw[6];
        for (int k = 0; k < 6; k++) fw[k] = r[k] * W[k];
        float Jw[6 * MAX_ACT];
        for (int k = 0; k < 6; k++)
            for (int a = 0; a < n_act; a++)
                Jw[k * n_act + a] = J[k * n_act + a] * W[k];

        float curr_err = 0.0f;
        for (int k = 0; k < 6; k++) curr_err += fw[k] * fw[k];

        // ── Jacobi column scaling ───────────────────────────────────────
        float col_scale[MAX_ACT];
        for (int a = 0; a < n_act; a++) {
            float sq = 0.0f;
            for (int k = 0; k < 6; k++) { float v = Jw[k*n_act+a]; sq += v*v; }
            col_scale[a] = sqrtf(sq) + 1e-8f;
        }
        float Js[6 * MAX_ACT];
        for (int k = 0; k < 6; k++)
            for (int a = 0; a < n_act; a++)
                Js[k * n_act + a] = Jw[k * n_act + a] / col_scale[a];

        // ── Normal equations + LM damping (float64) ────────────────────
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
            rhs_s[i] = -rb;
            A_s[i*n_act + i] += (double)lam;
        }

        // Mask fixed joints (zero row+col, unit diagonal, zero rhs).
        for (int a = 0; a < n_act; a++) {
            if (!s_fixed_mask[a]) continue;
            for (int j = 0; j < n_act; j++) A_s[a*n_act+j] = A_s[j*n_act+a] = 0.0;
            A_s[a*n_act+a] = 1.0;
            rhs_s[a] = 0.0;
        }

        chol_solve(A_s, rhs_s, n_act);

        // Unscale.
        float delta[MAX_ACT];
        for (int a = 0; a < n_act; a++)
            delta[a] = (float)rhs_s[a] / col_scale[a];

        // ── Trust-region step clipping ──────────────────────────────────
        {
            const float p_r = norm3(r);
            const float o_r = norm3(r + 3);
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

        // ── Line search over 5 step sizes ──────────────────────────────
        const float alphas[5] = { 1.0f, 0.5f, 0.25f, 0.1f, 0.025f };
        float best_alpha_err = 1e30f;
        int   best_alpha_idx = 0;
        float r_trial[6];

        for (int ai = 0; ai < 5; ai++) {
            float cfg_trial[MAX_ACT];
            for (int a = 0; a < n_act; a++)
                cfg_trial[a] = clampf(cfg[a] + alphas[ai] * delta[a],
                                      s_lower[a], s_upper[a]);

            compute_residual_only(
                cfg_trial, T_world,
                s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                s_mimic_mul, s_mimic_off, s_mimic_act_idx, s_topo_inv,
                s_target_T, target_jnt, n_joints, n_act, r_trial);

            float err_trial = 0.0f;
            for (int k = 0; k < 6; k++) {
                float rw = r_trial[k] * W[k];
                err_trial += rw * rw;
            }
            if (err_trial < best_alpha_err) {
                best_alpha_err = err_trial;
                best_alpha_idx = ai;
            }
        }

        // Compute winning trial configuration.
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

static ffi::Error LsIkCudaImpl(
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
    int64_t max_iter,
    float   pos_weight,
    float   ori_weight,
    float   lambda_init,
    float   eps_pos,
    float   eps_ori,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out_err)
{
    const int n_problems = static_cast<int>(target_T.dimensions()[0]);
    const int n_seeds    = static_cast<int>(seeds.dimensions()[1]);
    const int n_act      = static_cast<int>(seeds.dimensions()[2]);
    const int n_joints   = static_cast<int>(twists.dimensions()[0]);

    // LM is register/local-memory heavy; keep block size modest.
    constexpr int THREADS_MAX = 32;
    const int threads  = n_seeds < THREADS_MAX ? n_seeds : THREADS_MAX;
    const int blocks_x = (n_seeds + threads - 1) / threads;

    ls_ik_lm_kernel<<<dim3(blocks_x, n_problems), threads, 0, stream>>>(
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
        static_cast<int>(max_iter),
        pos_weight, ori_weight, lambda_init,
        eps_pos, eps_ori);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LsIkCudaFfi, LsIkCudaImpl,
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
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // ancestor_mask
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // target_T
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lower
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // upper
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // fixed_mask
        .Attr<int64_t>("target_jnt")
        .Attr<int64_t>("max_iter")
        .Attr<float>("pos_weight")
        .Attr<float>("ori_weight")
        .Attr<float>("lambda_init")
        .Attr<float>("eps_pos")
        .Attr<float>("eps_ori")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out cfgs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out errors
);
