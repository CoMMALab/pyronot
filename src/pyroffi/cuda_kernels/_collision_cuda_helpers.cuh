#pragma once

#include "_fk_cuda_helpers.cuh"

#include <cmath>

__device__ __forceinline__ float sql2(
    float ax, float ay, float az, float bx, float by, float bz)
{
    float dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float sphere_sphere_dist(
    float ax, float ay, float az, float ar,
    float bx, float by, float bz, float br)
{
    return sqrtf(sql2(ax, ay, az, bx, by, bz)) - (ar + br);
}

__device__ __forceinline__ float sphere_capsule_dist(
    float sx, float sy, float sz, float sr,
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr)
{
    float vx = x2 - x1, vy = y2 - y1, vz = z2 - z1;
    float len2 = vx * vx + vy * vy + vz * vz;
    float t = 0.0f;
    if (len2 > 1e-12f) {
        t = ((sx - x1) * vx + (sy - y1) * vy + (sz - z1) * vz) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    float cx = x1 + t * vx, cy = y1 + t * vy, cz = z1 + t * vz;
    return sqrtf(sql2(sx, sy, sz, cx, cy, cz)) - (sr + cr);
}

__device__ __forceinline__ float box_sdf_local(
    float p1, float p2, float p3,
    float hl1, float hl2, float hl3)
{
    float q1 = fabsf(p1) - hl1, q2 = fabsf(p2) - hl2, q3 = fabsf(p3) - hl3;
    float mq1 = fmaxf(q1, 0.0f), mq2 = fmaxf(q2, 0.0f), mq3 = fmaxf(q3, 0.0f);
    return sqrtf(mq1 * mq1 + mq2 * mq2 + mq3 * mq3)
           + fminf(fmaxf(fmaxf(q1, q2), q3), 0.0f);
}

__device__ __forceinline__ float sphere_box_dist(
    float sx, float sy, float sz, float sr,
    float bcx, float bcy, float bcz,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float a3x, float a3y, float a3z,
    float hl1, float hl2, float hl3)
{
    float dx = sx - bcx, dy = sy - bcy, dz = sz - bcz;
    return box_sdf_local(dx * a1x + dy * a1y + dz * a1z,
                         dx * a2x + dy * a2y + dz * a2z,
                         dx * a3x + dy * a3y + dz * a3z,
                         hl1, hl2, hl3) - sr;
}

__device__ __forceinline__ float capsule_box_dist(
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr,
    float bcx, float bcy, float bcz,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float a3x, float a3y, float a3z,
    float hl1, float hl2, float hl3)
{
    float d1x = x1 - bcx, d1y = y1 - bcy, d1z = z1 - bcz;
    float al1 = d1x * a1x + d1y * a1y + d1z * a1z;
    float al2 = d1x * a2x + d1y * a2y + d1z * a2z;
    float al3 = d1x * a3x + d1y * a3y + d1z * a3z;
    float d2x = x2 - bcx, d2y = y2 - bcy, d2z = z2 - bcz;
    float bl1 = d2x * a1x + d2y * a1y + d2z * a1z;
    float bl2 = d2x * a2x + d2y * a2y + d2z * a2z;
    float bl3 = d2x * a3x + d2y * a3y + d2z * a3z;
    float ab1 = bl1 - al1, ab2 = bl2 - al2, ab3 = bl3 - al3;
    float ab_len2 = ab1 * ab1 + ab2 * ab2 + ab3 * ab3;
    float t = 0.0f;
    if (ab_len2 > 1e-12f) {
        t = (-al1 * ab1 - al2 * ab2 - al3 * ab3) / ab_len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    return box_sdf_local(al1 + t * ab1, al2 + t * ab2, al3 + t * ab3,
                         hl1, hl2, hl3) - cr;
}

__device__ __forceinline__ float capsule_capsule_dist(
    float ax1, float ay1, float az1,
    float ax2, float ay2, float az2, float ar,
    float bx1, float by1, float bz1,
    float bx2, float by2, float bz2, float br)
{
    float d1x = ax2 - ax1, d1y = ay2 - ay1, d1z = az2 - az1;
    float d2x = bx2 - bx1, d2y = by2 - by1, d2z = bz2 - bz1;
    float rx = ax1 - bx1,  ry = ay1 - by1,  rz = az1 - bz1;
    float a = d1x * d1x + d1y * d1y + d1z * d1z;
    float e = d2x * d2x + d2y * d2y + d2z * d2z;
    float f = d2x * rx  + d2y * ry  + d2z * rz;
    const float EPS = 1e-10f;
    float s, t;
    if (a <= EPS && e <= EPS) {
        s = t = 0.0f;
    } else if (a <= EPS) {
        s = 0.0f;
        t = fmaxf(0.0f, fminf(1.0f, f / e));
    } else {
        float c = d1x * rx + d1y * ry + d1z * rz;
        if (e <= EPS) {
            t = 0.0f;
            s = fmaxf(0.0f, fminf(1.0f, -c / a));
        } else {
            float b = d1x * d2x + d1y * d2y + d1z * d2z;
            float denom = a * e - b * b;
            s = (fabsf(denom) > EPS) ? fmaxf(0.0f, fminf(1.0f, (b * f - c * e) / denom)) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = fmaxf(0.0f, fminf(1.0f, -c / a));
            } else if (t > 1.0f) {
                t = 1.0f;
                s = fmaxf(0.0f, fminf(1.0f, (b - c) / a));
            }
        }
    }
    float px = ax1 + s * d1x - (bx1 + t * d2x);
    float py = ay1 + s * d1y - (by1 + t * d2y);
    float pz = az1 + s * d1z - (bz1 + t * d2z);
    return sqrtf(px * px + py * py + pz * pz) - (ar + br);
}

__device__ __forceinline__ float sphere_halfspace_dist(
    float sx, float sy, float sz, float sr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    return (sx - px) * nx + (sy - py) * ny + (sz - pz) * nz - sr;
}

__device__ __forceinline__ float capsule_halfspace_dist(
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    float d1 = (x1 - px) * nx + (y1 - py) * ny + (z1 - pz) * nz;
    float d2 = (x2 - px) * nx + (y2 - py) * ny + (z2 - pz) * nz;
    return fminf(d1, d2) - cr;
}

__device__ __forceinline__ void apply_se3_point(
    const float* __restrict__ T,
    const float* __restrict__ p,
    float* __restrict__ out)
{
    quat_rotate(T, p, out);
    out[0] += T[4];
    out[1] += T[5];
    out[2] += T[6];
}

__device__ __forceinline__ float colldist_from_sdf(float d, float margin)
{
    d = fminf(d, margin);
    float val;
    if (d < 0.0f) {
        val = d - 0.5f * margin;
    } else {
        float diff = d - margin;
        val = -0.5f / (margin + 1e-6f) * diff * diff;
    }
    return fminf(val, 0.0f);
}
