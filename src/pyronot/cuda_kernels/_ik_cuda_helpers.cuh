/**
 * Shared IK helpers for CUDA kernels.
 *
 * Provides:
 *   - cross3, norm3, clampf
 *   - chol_solve (float64 Cholesky solve)
 *   - compute_residual_and_jacobian
 *   - compute_residual_only
 */

#pragma once

#include "_fk_cuda_helpers.cuh"

#include <cmath>

// ---------------------------------------------------------------------------
// Compile-time limits (override by defining before include)
// ---------------------------------------------------------------------------

#ifndef MAX_JOINTS
#define MAX_JOINTS 64
#endif

#ifndef MAX_ACT
#define MAX_ACT 16
#endif

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

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
// Cholesky solver (sequential, float64, in-place)
// ---------------------------------------------------------------------------

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
