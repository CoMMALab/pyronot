/**
 * Shared IK helpers for CUDA kernels.
 *
 * Provides:
 *   - xorshift32, rng_normal        (fast GPU PRNG, Box–Muller)
 *   - cross3, norm3, clampf
 *   - lbfgs_two_loop                (Nocedal two-loop L-BFGS recursion)
 *   - chol_solve                    (float32/float64 Cholesky solve)
 *   - compute_residual_and_jacobian
 *   - compute_residual_only
 *   - compute_multi_ee_residual_and_jacobian
 *   - compute_multi_ee_residual_only
 */

#pragma once

#include "_fk_cuda_helpers.cuh"

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// Compile-time limits (override by defining before include)
// ---------------------------------------------------------------------------

#ifndef MAX_JOINTS
#define MAX_JOINTS 64
#endif

#ifndef MAX_ACT
#define MAX_ACT 16
#endif

#ifndef MAX_EE
#define MAX_EE 4
#endif

#ifndef MAX_PARTICLES
#define MAX_PARTICLES 32
#endif

#ifndef MAX_LBFGS_M
#define MAX_LBFGS_M 8
#endif

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

// xorshift32 fast PRNG (GPU-friendly, no libcurand dependency).
static __device__ __forceinline__
uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Box–Muller transform: returns one standard-normal sample, advances state twice.
static __device__ __forceinline__
float rng_normal(uint32_t& state)
{
    // Two uniform samples in (0, 1] via bit-masking to 24-bit mantissa.
    const float u1 = fmaxf(1e-7f, (float)(xorshift32(state) >> 8) * (1.0f / 16777216.0f));
    const float u2 =              (float)(xorshift32(state) >> 8) * (1.0f / 16777216.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static __device__ __forceinline__
void cross3(const float* __restrict__ a,
            const float* __restrict__ b,
            float* __restrict__ out)
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static __device__ __forceinline__
float norm3(const float* __restrict__ v)
{
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static __device__ __forceinline__
float clampf(float x, float lo, float hi)
{
    return fmaxf(lo, fminf(hi, x));
}

// ---------------------------------------------------------------------------
// L-BFGS two-loop recursion (float32, in-place on p)
// ---------------------------------------------------------------------------

/**
 * Compute the L-BFGS descent direction  p = H_k^{-1} (-g)  using the
 * Nocedal two-loop recursion.
 *
 * Initialises q = -g, applies m history pairs (newest → oldest), scales by
 * the Shanno–Kettler γ, then applies the second loop (oldest → newest).
 * Result stored in p[].
 *
 * @param g           gradient at current point (n_act,)
 * @param s_buf       circular buffer of s = Δq vectors  (m_max, n_act)
 * @param y_buf       circular buffer of y = Δg vectors  (m_max, n_act)
 * @param rho_buf     rho = 1/(y^T s) scalars            (m_max,)
 * @param alpha_buf   scratch space                       (m_max,)
 * @param n_act       number of active DOF
 * @param m_max       buffer capacity (≤ MAX_LBFGS_M)
 * @param m_used      number of valid pairs stored        (0 → m_max)
 * @param newest      index of the most recently stored pair (-1 if m_used==0)
 * @param p           output: descent direction (n_act,)
 */
static __device__ void lbfgs_two_loop(
    const float* __restrict__ g,
    const float* __restrict__ s_buf,
    const float* __restrict__ y_buf,
    const float* __restrict__ rho_buf,
    float*       __restrict__ alpha_buf,
    int n_act, int m_max, int m_used, int newest,
    float* __restrict__ p)
{
    // q = -g
    for (int a = 0; a < n_act; a++) p[a] = -g[a];

    if (m_used == 0) return;   // No history → steepest descent.

    // First loop: newest → oldest
    for (int i = 0; i < m_used; i++) {
        const int idx        = (newest - i + m_max) % m_max;
        const float* s_i     = s_buf + idx * n_act;
        const float* y_i     = y_buf + idx * n_act;
        const float  rho_i   = rho_buf[idx];
        float alpha_i = 0.0f;
        for (int a = 0; a < n_act; a++) alpha_i += rho_i * s_i[a] * p[a];
        alpha_buf[i] = alpha_i;
        for (int a = 0; a < n_act; a++) p[a] -= alpha_i * y_i[a];
    }

    // Initial Hessian: H_0 = γ I  (Shanno–Kettler scaling using newest pair).
    {
        const float* s_k = s_buf + newest * n_act;
        const float* y_k = y_buf + newest * n_act;
        float sy = 0.0f, yy = 0.0f;
        for (int a = 0; a < n_act; a++) { sy += s_k[a] * y_k[a]; yy += y_k[a] * y_k[a]; }
        const float gamma = (yy > 1e-20f) ? fmaxf(1e-8f, fminf(1.0f, sy / yy)) : 1.0f;
        for (int a = 0; a < n_act; a++) p[a] *= gamma;
    }

    // Second loop: oldest → newest
    for (int i = m_used - 1; i >= 0; i--) {
        const int idx      = (newest - i + m_max) % m_max;
        const float* s_i   = s_buf + idx * n_act;
        const float* y_i   = y_buf + idx * n_act;
        const float  rho_i = rho_buf[idx];
        float beta = 0.0f;
        for (int a = 0; a < n_act; a++) beta += rho_i * y_i[a] * p[a];
        const float coeff = alpha_buf[i] - beta;
        for (int a = 0; a < n_act; a++) p[a] += s_i[a] * coeff;
    }
    // p = H_k^{-1} (-g) = descent direction.
}

// ---------------------------------------------------------------------------
// Cholesky solver (sequential, in-place)
// ---------------------------------------------------------------------------

static __device__ bool chol_solve(float* __restrict__ A,
                                  float* __restrict__ b,
                                  int n)
{
    for (int k = 0; k < n; k++) {
        float s = A[k*n + k];
        for (int p = 0; p < k; p++) { float lkp = A[k*n+p]; s -= lkp*lkp; }
        if (s <= 0.0f) {
            for (int i = 0; i < n; i++) b[i] = 0.0f;
            return false;
        }
        float lkk = sqrtf(s);
        A[k*n + k] = lkk;
        for (int i = k+1; i < n; i++) {
            float t = A[i*n + k];
            for (int p = 0; p < k; p++) t -= A[i*n+p] * A[k*n+p];
            A[i*n + k] = t / lkk;
        }
        for (int j = k+1; j < n; j++) A[k*n + j] = 0.0f;
    }

    // Forward substitution: L y = b
    float y[MAX_ACT];
    for (int i = 0; i < n; i++) {
        float s = b[i];
        for (int p = 0; p < i; p++) s -= A[i*n+p] * y[p];
        y[i] = s / A[i*n + i];
    }

    // Backward substitution: L^T x = y
    for (int i = n-1; i >= 0; i--) {
        float s = y[i];
        for (int p = i+1; p < n; p++) s -= A[p*n + i] * b[p];
        b[i] = s / A[i*n + i];
    }
    return true;
}

static __device__ bool chol_solve(double* __restrict__ A,
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

static __device__ void compute_residual_and_jacobian(
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

// ---------------------------------------------------------------------------
// Residual-only evaluation (no Jacobian)
// ---------------------------------------------------------------------------

static __device__ void compute_residual_only(
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
// Multi-EE IK residual and stacked geometric Jacobian
// ---------------------------------------------------------------------------

/**
 * Compute stacked residuals r[6*n_ee] and stacked Jacobian J[6*n_ee*n_act]
 * for n_ee end-effectors simultaneously using a single FK call.
 *
 * Layout: r = [r_0 | r_1 | ... | r_{n_ee-1}], each r_i is 6 elements.
 *         J = [J_0 | J_1 | ... | J_{n_ee-1}], each J_i is (6, n_act) row-major.
 *
 * @param target_jnts     (n_ee,)             joint index for each EE
 * @param ancestor_masks  (n_ee, n_joints)    row-major ancestor bitmask per EE
 * @param target_Ts       (n_ee, 7)           row-major target poses per EE [w,x,y,z,tx,ty,tz]
 */
static __device__ void compute_multi_ee_residual_and_jacobian(
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
    const int*   __restrict__ target_jnts,     // (n_ee,)
    const int*   __restrict__ ancestor_masks,  // (n_ee, n_joints) row-major
    const float* __restrict__ target_Ts,       // (n_ee, 7) row-major
    int n_joints, int n_act, int n_ee,
    float* __restrict__ r,   // output: (6 * n_ee)
    float* __restrict__ J)   // output: (6 * n_ee * n_act), zeroed then filled
{
    // Single FK call for all EEs.
    fk_single(cfg, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv,
              T_world, n_joints, n_act);

    // Zero Jacobian.
    for (int i = 0; i < 6 * n_ee * n_act; i++) J[i] = 0.0f;

    // Compute EE positions and residuals; store arm vectors for Jacobian.
    float arm[MAX_EE * 3];
    for (int ee = 0; ee < n_ee; ee++) {
        const int    tgt_jnt  = target_jnts[ee];
        const float* target_T = target_Ts + ee * 7;
        const float* T_ee_w   = T_world + tgt_jnt * 7;
        const float p_ee[3] = { T_ee_w[4], T_ee_w[5], T_ee_w[6] };
        const float q_ee[4] = { T_ee_w[0], T_ee_w[1], T_ee_w[2], T_ee_w[3] };

        const float p_tgt[3] = { target_T[4], target_T[5], target_T[6] };
        const float q_tgt[4] = { target_T[0], target_T[1], target_T[2], target_T[3] };

        float* r_ee = r + ee * 6;
        r_ee[0] = p_ee[0] - p_tgt[0];
        r_ee[1] = p_ee[1] - p_tgt[1];
        r_ee[2] = p_ee[2] - p_tgt[2];

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
            r_ee[3] = q_err[1] * inv_sin;
            r_ee[4] = q_err[2] * inv_sin;
            r_ee[5] = q_err[3] * inv_sin;
        } else {
            r_ee[3] = 2.0f * q_err[1];
            r_ee[4] = 2.0f * q_err[2];
            r_ee[5] = 2.0f * q_err[3];
        }

        arm[ee*3+0] = p_ee[0];
        arm[ee*3+1] = p_ee[1];
        arm[ee*3+2] = p_ee[2];
    }

    // Build stacked Jacobian: one pass over joints, inner loop over EEs.
    for (int j = 0; j < n_joints; j++) {
        int a1 = act_idx[j];
        int a2 = mimic_act_idx[j];
        if (a1 < 0 && a2 < 0) continue;

        const float* tw = twists + j * 6;
        const float ang_sq = tw[3]*tw[3] + tw[4]*tw[4] + tw[5]*tw[5];
        const float lin_sq = tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2];

        const float* T_j = T_world + j * 7;
        const float* q_j = T_j;
        const float* p_j = T_j + 4;

        bool is_revolute;
        float z_j[3];
        if (ang_sq > 1e-6f) {
            is_revolute = true;
            const float inv_ang = 1.0f / sqrtf(ang_sq);
            const float body_ax[3] = { tw[3]*inv_ang, tw[4]*inv_ang, tw[5]*inv_ang };
            quat_rotate(q_j, body_ax, z_j);
        } else if (lin_sq > 1e-6f) {
            is_revolute = false;
            const float inv_lin = 1.0f / sqrtf(lin_sq);
            const float body_ax[3] = { tw[0]*inv_lin, tw[1]*inv_lin, tw[2]*inv_lin };
            quat_rotate(q_j, body_ax, z_j);
        } else {
            continue;
        }

        for (int ee = 0; ee < n_ee; ee++) {
            if (!ancestor_masks[ee * n_joints + j]) continue;

            float* J_ee = J + ee * 6 * n_act;
            const float* arm_ee = arm + ee * 3;

            float jg_lin[3], jg_ang[3];
            if (is_revolute) {
                const float arm_j[3] = { arm_ee[0]-p_j[0], arm_ee[1]-p_j[1], arm_ee[2]-p_j[2] };
                cross3(z_j, arm_j, jg_lin);
                jg_ang[0] = z_j[0]; jg_ang[1] = z_j[1]; jg_ang[2] = z_j[2];
            } else {
                jg_lin[0] = z_j[0]; jg_lin[1] = z_j[1]; jg_lin[2] = z_j[2];
                jg_ang[0] = 0.0f;   jg_ang[1] = 0.0f;   jg_ang[2] = 0.0f;
            }

            if (a1 >= 0) {
                const float s = mimic_mul[j];
                J_ee[0*n_act + a1] += s * jg_lin[0];
                J_ee[1*n_act + a1] += s * jg_lin[1];
                J_ee[2*n_act + a1] += s * jg_lin[2];
                J_ee[3*n_act + a1] += s * jg_ang[0];
                J_ee[4*n_act + a1] += s * jg_ang[1];
                J_ee[5*n_act + a1] += s * jg_ang[2];
            }
            if (a2 >= 0) {
                const float s = mimic_mul[j];
                J_ee[0*n_act + a2] += s * jg_lin[0];
                J_ee[1*n_act + a2] += s * jg_lin[1];
                J_ee[2*n_act + a2] += s * jg_lin[2];
                J_ee[3*n_act + a2] += s * jg_ang[0];
                J_ee[4*n_act + a2] += s * jg_ang[1];
                J_ee[5*n_act + a2] += s * jg_ang[2];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-EE residual-only evaluation (no Jacobian)
// ---------------------------------------------------------------------------

/**
 * Compute stacked residuals r[6*n_ee] for n_ee end-effectors using one FK call.
 */
static __device__ void compute_multi_ee_residual_only(
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
    const int*   __restrict__ target_jnts,  // (n_ee,)
    const float* __restrict__ target_Ts,    // (n_ee, 7) row-major
    int n_joints, int n_act, int n_ee,
    float* __restrict__ r)   // output: (6 * n_ee)
{
    fk_single(cfg, twists, parent_tf, parent_idx, act_idx,
              mimic_mul, mimic_off, mimic_act_idx, topo_inv,
              T_world, n_joints, n_act);

    for (int ee = 0; ee < n_ee; ee++) {
        const int    tgt_jnt  = target_jnts[ee];
        const float* target_T = target_Ts + ee * 7;
        const float* T_ee_w   = T_world + tgt_jnt * 7;
        const float p_ee[3] = { T_ee_w[4], T_ee_w[5], T_ee_w[6] };
        const float q_ee[4] = { T_ee_w[0], T_ee_w[1], T_ee_w[2], T_ee_w[3] };

        const float p_tgt[3] = { target_T[4], target_T[5], target_T[6] };
        const float q_tgt[4] = { target_T[0], target_T[1], target_T[2], target_T[3] };

        float* r_ee = r + ee * 6;
        r_ee[0] = p_ee[0] - p_tgt[0];
        r_ee[1] = p_ee[1] - p_tgt[1];
        r_ee[2] = p_ee[2] - p_tgt[2];

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
            r_ee[3] = q_err[1] * inv_sin;
            r_ee[4] = q_err[2] * inv_sin;
            r_ee[5] = q_err[3] * inv_sin;
        } else {
            r_ee[3] = 2.0f * q_err[1];
            r_ee[4] = 2.0f * q_err[2];
            r_ee[5] = 2.0f * q_err[3];
        }
    }
}
