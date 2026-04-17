/**
 * Shared SE(3) math helpers and single-thread FK device function.
 *
 * Included by both _fk_cuda_kernel.cu and _hjcd_ik_cuda_kernel.cu so the FK
 * logic is defined exactly once and the IK kernel can call fk_single()
 * directly without duplicating code.
 *
 * Tangent convention (jaxlie SE3):
 *   tangent = [v_x, v_y, v_z, omega_x, omega_y, omega_z]
 *   i.e., linear velocity first, angular velocity last.
 *
 * SE(3) storage: quaternion + translation, [w, x, y, z, tx, ty, tz].
 */

#pragma once

#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// SE(3) math helpers
// ---------------------------------------------------------------------------

/**
 * Hamilton quaternion product.
 * q1, q2, out are all in wxyz order.
 */
__device__ __forceinline__
void quat_mul(const float* __restrict__ q1,
              const float* __restrict__ q2,
              float* __restrict__ out)
{
    const float w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    const float w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
    out[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    out[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    out[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    out[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

/**
 * Rotate vector v by unit quaternion q (wxyz). Result written to out.
 * Uses the cross-product form:  v' = v + 2w*(q_vec x v) + 2*(q_vec x (q_vec x v))
 */
__device__ __forceinline__
void quat_rotate(const float* __restrict__ q,
                 const float* __restrict__ v,
                 float* __restrict__ out)
{
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    const float cx = y*v[2] - z*v[1];
    const float cy = z*v[0] - x*v[2];
    const float cz = x*v[1] - y*v[0];
    const float ccx = y*cz - z*cy;
    const float ccy = z*cx - x*cz;
    const float ccz = x*cy - y*cx;
    out[0] = v[0] + 2.0f*w*cx + 2.0f*ccx;
    out[1] = v[1] + 2.0f*w*cy + 2.0f*ccy;
    out[2] = v[2] + 2.0f*w*cz + 2.0f*ccz;
}

/**
 * SE(3) composition:  T_out = T1 @ T2.
 * Layout: [w, x, y, z, tx, ty, tz]
 */
__device__ __forceinline__
void se3_compose(const float* __restrict__ T1,
                 const float* __restrict__ T2,
                 float* __restrict__ T_out)
{
    quat_mul(T1, T2, T_out);
    float rotated[3];
    quat_rotate(T1, T2 + 4, rotated);
    T_out[4] = rotated[0] + T1[4];
    T_out[5] = rotated[1] + T1[5];
    T_out[6] = rotated[2] + T1[6];
}

/**
 * SE(3) exponential map.
 *
 * Tangent convention (jaxlie):
 *   tangent[0:3] = v      (linear)
 *   tangent[3:6] = omega  (angular)
 *
 * Output: [w, x, y, z, tx, ty, tz]
 */
__device__ __forceinline__
void se3_exp(const float* __restrict__ tangent,
             float* __restrict__ T_out)
{
    constexpr float EPS = 1e-6f;
    constexpr float EPS2 = EPS * EPS;

    const float vx = tangent[0], vy = tangent[1], vz = tangent[2];
    const float ox = tangent[3], oy = tangent[4], oz = tangent[5];

    const float theta2 = ox*ox + oy*oy + oz*oz;
    if (theta2 < EPS2) {
        T_out[0] = 1.0f; T_out[1] = 0.0f; T_out[2] = 0.0f; T_out[3] = 0.0f;
        T_out[4] = vx;   T_out[5] = vy;   T_out[6] = vz;
    } else {
        const float theta = sqrtf(theta2);
        const float half_theta = theta * 0.5f;
        float sin_half, cos_half;
        sincosf(half_theta, &sin_half, &cos_half);
        const float s = sin_half / theta;
        T_out[0] = cos_half;
        T_out[1] = s * ox;
        T_out[2] = s * oy;
        T_out[3] = s * oz;

        float sin_t, cos_t;
        sincosf(theta, &sin_t, &cos_t);
        const float A = sin_t / theta;
        const float B = (1.0f - cos_t) / theta2;
        const float C = (theta - sin_t) / (theta2 * theta);

        const float omv_x = oy*vz - oz*vy;
        const float omv_y = oz*vx - ox*vz;
        const float omv_z = ox*vy - oy*vx;

        const float om_dot_v = ox*vx + oy*vy + oz*vz;
        const float om2v_x = ox*om_dot_v - theta2*vx;
        const float om2v_y = oy*om_dot_v - theta2*vy;
        const float om2v_z = oz*om_dot_v - theta2*vz;

        T_out[4] = A*vx + B*omv_x + C*om2v_x;
        T_out[5] = A*vy + B*omv_y + C*om2v_y;
        T_out[6] = A*vz + B*omv_z + C*om2v_z;
    }
}

// ---------------------------------------------------------------------------
// Single-thread FK
// ---------------------------------------------------------------------------

/**
 * Run forward kinematics for a single joint configuration.
 *
 * Mirrors the body of fk_kernel exactly.  Call this from any device
 * function that needs FK without launching a new kernel (e.g., the IK
 * optimization loops).
 *
 * @param cfg            (n_act,)          actuated configuration
 * @param twists         (n_joints, 6)     Lie-algebra twist per joint
 * @param parent_tf      (n_joints, 7)     constant T_parent_joint [wxyz_xyz]
 * @param parent_idx     (n_joints,)       parent joint index, -1 for roots
 * @param act_idx        (n_joints,)       actuated source index, -1 if fixed
 * @param mimic_mul      (n_joints,)       mimic multiplier (1.0 for non-mimic)
 * @param mimic_off      (n_joints,)       mimic offset (0.0 for non-mimic)
 * @param mimic_act_idx  (n_joints,)       mimicked actuated idx, -1 if not mimic
 * @param topo_inv       (n_joints,)       sorted_i -> orig_j mapping
 * @param T_world        (n_joints, 7)     output world transforms [wxyz_xyz]
 * @param n_joints       total joint count
 * @param n_act          actuated joint count
 */
__device__ void fk_single(
    const float* __restrict__ cfg,
    const float* __restrict__ twists,
    const float* __restrict__ parent_tf,
    const int*   __restrict__ parent_idx,
    const int*   __restrict__ act_idx,
    const float* __restrict__ mimic_mul,
    const float* __restrict__ mimic_off,
    const int*   __restrict__ mimic_act_idx,
    const int*   __restrict__ topo_inv,
    float*       __restrict__ T_world,
    int n_joints, int n_act)
{
    for (int i = 0; i < n_joints; ++i) {
        const int j = topo_inv[i];

        // Expand actuated config to full joint value q_j.
        const int m_idx  = mimic_act_idx[j];
        const int a_idx  = act_idx[j];
        const int src    = (m_idx != -1) ? m_idx : a_idx;
        const float q_ref = (src == -1) ? 0.0f : cfg[src];
        const float q_j   = q_ref * mimic_mul[j] + mimic_off[j];

        // T_parent_child = parent_tf[j] @ delta_T.
        float T_pc[7];
        if (src == -1 && mimic_off[j] == 0.0f) {
            #pragma unroll
            for (int k = 0; k < 7; ++k) T_pc[k] = parent_tf[j * 7 + k];
        } else {
            // tangent = twists[j] * q_j,  then delta_T = SE3.exp(tangent).
            float tangent[6];
            #pragma unroll
            for (int k = 0; k < 6; ++k)
                tangent[k] = twists[j * 6 + k] * q_j;

            float delta_T[7];
            se3_exp(tangent, delta_T);
            se3_compose(parent_tf + j * 7, delta_T, T_pc);
        }

        // T_world[j] = T_world[parent_idx[j]] @ T_pc   (root: = T_pc).
        float* dst = T_world + j * 7;
        const int p = parent_idx[j];
        if (p == -1) {
            #pragma unroll
            for (int k = 0; k < 7; ++k) dst[k] = T_pc[k];
        } else {
            se3_compose(T_world + p * 7, T_pc, dst);
        }
    }
}
