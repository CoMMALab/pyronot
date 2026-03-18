/**
 * Batched collision-distance CUDA kernels for pyronot, with XLA/JAX FFI bindings.
 *
 * Inspired by pRRTC (https://github.com/lyf44/pRRTC).
 *
 * Performance design:
 *   - Type-split world kernels: one kernel per obstacle type (sphere/capsule/box/halfspace).
 *     No per-thread branching on obstacle type; each kernel is type-homogeneous.
 *   - Shared-memory tiling: each block loads TILE_M obstacles into shared memory,
 *     then all BLOCK_K threads compute distances against the tile.
 *   - Fused S-reduction (sphere-robot only): `world_sphere_reduced_vs_*` kernels
 *     output [B, N, M] directly by reducing S spheres per link inside CUDA,
 *     eliminating the Python-side reshape + min.
 *   - __launch_bounds__(256, 2) on every kernel hints the compiler to bound
 *     register pressure so at least 2 blocks fit per SM.
 *
 * World geometry types supported:
 *   Sphere [Ms, 4]:     (x, y, z, r)
 *   Capsule [Mc, 7]:    (x1, y1, z1, x2, y2, z2, r)
 *   Box [Mb, 15]:       (cx, cy, cz, a1x..a1z, a2x..a2z, a3x..a3z, hl1, hl2, hl3)
 *   HalfSpace [Mh, 6]:  (nx, ny, nz, px, py, pz)  — outward normal + point on plane
 *
 * Robot geometry (SoA layout for coalesced loads):
 *   Sphere-robot:  centers [3, B, K]  (component-major; K = S * N_links)
 *                  radii   [B, K]
 *   Capsule-robot: caps    [7, B, N]  (component-major)
 *
 * Grid conventions:
 *   World kernels — blockIdx.x = batch b
 *                   blockIdx.y = tile over robot dimension (K or N)
 *                   blockIdx.z = tile over obstacle dimension (M_type / TILE_M)
 *   Self kernels  — 1-D grid, one thread per (batch b, active pair p)
 *
 * Distance convention (matches pyronot JAX backend):
 *   positive → separated
 *   negative → penetration
 *
 * Build: bash src/pyronot/cuda_kernels/build_collision_cuda.sh
 */

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ── Grid / tile constants ─────────────────────────────────────────────────────

/// Threads per block along the robot (K or N) dimension.
constexpr int BLOCK_K = 256;

/// World obstacles loaded into shared memory per tile.
constexpr int TILE_M  = 16;

// ── Math helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__ float sql2(
    float ax, float ay, float az, float bx, float by, float bz)
{
    float dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx*dx + dy*dy + dz*dz;
}

// ── Primitive signed-distance functions ───────────────────────────────────────

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
    float vx = x2-x1, vy = y2-y1, vz = z2-z1;
    float len2 = vx*vx + vy*vy + vz*vz;
    float t = 0.0f;
    if (len2 > 1e-12f) {
        t = ((sx-x1)*vx + (sy-y1)*vy + (sz-z1)*vz) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    float cx = x1+t*vx, cy = y1+t*vy, cz = z1+t*vz;
    return sqrtf(sql2(sx, sy, sz, cx, cy, cz)) - (sr + cr);
}

__device__ __forceinline__ float box_sdf_local(
    float p1, float p2, float p3,
    float hl1, float hl2, float hl3)
{
    float q1 = fabsf(p1)-hl1, q2 = fabsf(p2)-hl2, q3 = fabsf(p3)-hl3;
    float mq1 = fmaxf(q1, 0.0f), mq2 = fmaxf(q2, 0.0f), mq3 = fmaxf(q3, 0.0f);
    return sqrtf(mq1*mq1 + mq2*mq2 + mq3*mq3) + fminf(fmaxf(fmaxf(q1,q2),q3), 0.0f);
}

__device__ __forceinline__ float sphere_box_dist(
    float sx, float sy, float sz, float sr,
    float bcx, float bcy, float bcz,
    float a1x, float a1y, float a1z,
    float a2x, float a2y, float a2z,
    float a3x, float a3y, float a3z,
    float hl1, float hl2, float hl3)
{
    float dx = sx-bcx, dy = sy-bcy, dz = sz-bcz;
    return box_sdf_local(dx*a1x+dy*a1y+dz*a1z,
                         dx*a2x+dy*a2y+dz*a2z,
                         dx*a3x+dy*a3y+dz*a3z,
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
    float d1x=x1-bcx, d1y=y1-bcy, d1z=z1-bcz;
    float al1=d1x*a1x+d1y*a1y+d1z*a1z;
    float al2=d1x*a2x+d1y*a2y+d1z*a2z;
    float al3=d1x*a3x+d1y*a3y+d1z*a3z;
    float d2x=x2-bcx, d2y=y2-bcy, d2z=z2-bcz;
    float bl1=d2x*a1x+d2y*a1y+d2z*a1z;
    float bl2=d2x*a2x+d2y*a2y+d2z*a2z;
    float bl3=d2x*a3x+d2y*a3y+d2z*a3z;
    float ab1=bl1-al1, ab2=bl2-al2, ab3=bl3-al3;
    float ab_len2 = ab1*ab1 + ab2*ab2 + ab3*ab3;
    float t = 0.0f;
    if (ab_len2 > 1e-12f) {
        t = (-al1*ab1 - al2*ab2 - al3*ab3) / ab_len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
    }
    return box_sdf_local(al1+t*ab1, al2+t*ab2, al3+t*ab3, hl1, hl2, hl3) - cr;
}

__device__ __forceinline__ float capsule_capsule_dist(
    float ax1, float ay1, float az1,
    float ax2, float ay2, float az2, float ar,
    float bx1, float by1, float bz1,
    float bx2, float by2, float bz2, float br)
{
    float d1x=ax2-ax1, d1y=ay2-ay1, d1z=az2-az1;
    float d2x=bx2-bx1, d2y=by2-by1, d2z=bz2-bz1;
    float rx=ax1-bx1,  ry=ay1-by1,  rz=az1-bz1;
    float a = d1x*d1x + d1y*d1y + d1z*d1z;
    float e = d2x*d2x + d2y*d2y + d2z*d2z;
    float f = d2x*rx  + d2y*ry  + d2z*rz;
    const float EPS = 1e-10f;
    float s, t;
    if (a <= EPS && e <= EPS) { s = t = 0.0f; }
    else if (a <= EPS) { s = 0.0f; t = fmaxf(0.0f, fminf(1.0f, f/e)); }
    else {
        float c = d1x*rx + d1y*ry + d1z*rz;
        if (e <= EPS) { t = 0.0f; s = fmaxf(0.0f, fminf(1.0f, -c/a)); }
        else {
            float b     = d1x*d2x + d1y*d2y + d1z*d2z;
            float denom = a*e - b*b;
            s = (fabsf(denom) > EPS) ? fmaxf(0.0f, fminf(1.0f, (b*f-c*e)/denom)) : 0.0f;
            t = (b*s + f) / e;
            if      (t < 0.0f) { t = 0.0f; s = fmaxf(0.0f, fminf(1.0f, -c/a)); }
            else if (t > 1.0f) { t = 1.0f; s = fmaxf(0.0f, fminf(1.0f, (b-c)/a)); }
        }
    }
    float px = ax1+s*d1x - (bx1+t*d2x);
    float py = ay1+s*d1y - (by1+t*d2y);
    float pz = az1+s*d1z - (bz1+t*d2z);
    return sqrtf(px*px + py*py + pz*pz) - (ar + br);
}

/** Sphere–halfspace signed distance.
 *  HalfSpace encoded as (nx,ny,nz, px,py,pz): unit outward normal + point on plane.
 *  dist = dot(sphere_center - plane_point, normal) - sphere_radius
 */
__device__ __forceinline__ float sphere_halfspace_dist(
    float sx, float sy, float sz, float sr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    return (sx-px)*nx + (sy-py)*ny + (sz-pz)*nz - sr;
}

/** Capsule–halfspace signed distance.
 *  Both endpoints evaluated; the nearer to the plane determines the distance.
 *  dist = min(dot(A-p,n), dot(B-p,n)) - capsule_radius
 */
__device__ __forceinline__ float capsule_halfspace_dist(
    float x1, float y1, float z1,
    float x2, float y2, float z2, float cr,
    float nx, float ny, float nz,
    float px, float py, float pz)
{
    float d1 = (x1-px)*nx + (y1-py)*ny + (z1-pz)*nz;
    float d2 = (x2-px)*nx + (y2-py)*ny + (z2-pz)*nz;
    return fminf(d1, d2) - cr;
}

// ── World collision: sphere-robot, non-reduced (output [B, K, M]) ─────────────
//
// Grid:  (B, ceil(K/BLOCK_K), ceil(M_type/TILE_M))
// Block: BLOCK_K threads
// Shared: TILE_M * obstacle_stride floats
// Each kernel writes to out[b, k, m_offset .. m_offset+M_type-1].

static __global__ __launch_bounds__(256, 2)
void wcs_vs_spheres(
    const float* __restrict__ sc,   // [3, B, K] SoA
    const float* __restrict__ sr,   // [B, K]
    const float* __restrict__ wd,   // [Ms, 4]
    float*       __restrict__ out,  // [B, K, M_total]
    int B, int K, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M;
    const int te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 4; i += BLOCK_K)
        sm[i] = wd[ms * 4 + i];
    __syncthreads();

    const int b = blockIdx.x, k = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || k >= K) return;
    const int bk = b*K + k;
    const float rad = sr[bk];
    if (rad < 0.0f) {
        for (int t = 0; t < te; t++) out[bk*M + mo+ms+t] = 1e9f;
        return;
    }
    const int BK = B*K;
    const float sx = sc[0*BK+bk], sy = sc[1*BK+bk], sz = sc[2*BK+bk];
    for (int t = 0; t < te; t++)
        out[bk*M + mo+ms+t] = sphere_sphere_dist(sx, sy, sz, rad,
            sm[t*4], sm[t*4+1], sm[t*4+2], sm[t*4+3]);
}

static __global__ __launch_bounds__(256, 2)
void wcs_vs_capsules(
    const float* __restrict__ sc, const float* __restrict__ sr,
    const float* __restrict__ wd, float* __restrict__ out,
    int B, int K, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 7; i += BLOCK_K) sm[i] = wd[ms*7 + i];
    __syncthreads();

    const int b = blockIdx.x, k = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || k >= K) return;
    const int bk = b*K + k; const float rad = sr[bk];
    if (rad < 0.0f) { for (int t=0;t<te;t++) out[bk*M+mo+ms+t]=1e9f; return; }
    const int BK = B*K;
    const float sx=sc[0*BK+bk], sy=sc[1*BK+bk], sz=sc[2*BK+bk];
    for (int t = 0; t < te; t++)
        out[bk*M+mo+ms+t] = sphere_capsule_dist(sx, sy, sz, rad,
            sm[t*7], sm[t*7+1], sm[t*7+2],
            sm[t*7+3], sm[t*7+4], sm[t*7+5], sm[t*7+6]);
}

static __global__ __launch_bounds__(256, 2)
void wcs_vs_boxes(
    const float* __restrict__ sc, const float* __restrict__ sr,
    const float* __restrict__ wd, float* __restrict__ out,
    int B, int K, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 15; i += BLOCK_K) sm[i] = wd[ms*15 + i];
    __syncthreads();

    const int b = blockIdx.x, k = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || k >= K) return;
    const int bk = b*K + k; const float rad = sr[bk];
    if (rad < 0.0f) { for (int t=0;t<te;t++) out[bk*M+mo+ms+t]=1e9f; return; }
    const int BK = B*K;
    const float sx=sc[0*BK+bk], sy=sc[1*BK+bk], sz=sc[2*BK+bk];
    for (int t = 0; t < te; t++) {
        const float* o = sm + t*15;
        out[bk*M+mo+ms+t] = sphere_box_dist(sx, sy, sz, rad,
            o[0],o[1],o[2], o[3],o[4],o[5], o[6],o[7],o[8], o[9],o[10],o[11],
            o[12], o[13], o[14]);
    }
}

static __global__ __launch_bounds__(256, 2)
void wcs_vs_halfspaces(
    const float* __restrict__ sc, const float* __restrict__ sr,
    const float* __restrict__ wd, float* __restrict__ out,
    int B, int K, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 6; i += BLOCK_K) sm[i] = wd[ms*6 + i];
    __syncthreads();

    const int b = blockIdx.x, k = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || k >= K) return;
    const int bk = b*K + k; const float rad = sr[bk];
    if (rad < 0.0f) { for (int t=0;t<te;t++) out[bk*M+mo+ms+t]=1e9f; return; }
    const int BK = B*K;
    const float sx=sc[0*BK+bk], sy=sc[1*BK+bk], sz=sc[2*BK+bk];
    for (int t = 0; t < te; t++)
        out[bk*M+mo+ms+t] = sphere_halfspace_dist(sx, sy, sz, rad,
            sm[t*6], sm[t*6+1], sm[t*6+2], sm[t*6+3], sm[t*6+4], sm[t*6+5]);
}

// ── World collision: sphere-robot, fused S-reduction (output [B, N, M]) ───────
//
// Grid:  (B, ceil(N/BLOCK_K), ceil(M_type/TILE_M))   — tiles over N links, not K spheres
// Each thread: computes min-distance over s=0..S-1 for one (b, n, m) tuple.
// K = S * N; sphere k for link n at sphere-index s is k = s*N + n.

static __global__ __launch_bounds__(256, 2)
void wcsr_vs_spheres(
    const float* __restrict__ sc,   // [3, B, K] SoA, K = S*N
    const float* __restrict__ sr,   // [B, K]
    const float* __restrict__ wd,   // [Ms, 4]
    float*       __restrict__ out,  // [B, N, M_total]
    int B, int K, int S, int N, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 4; i += BLOCK_K) sm[i] = wd[ms*4 + i];
    __syncthreads();

    const int b = blockIdx.x, n = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || n >= N) return;

    const int BK = B*K;
    float min_d[TILE_M];
    #pragma unroll
    for (int t = 0; t < TILE_M; t++) min_d[t] = 1e9f;

    for (int s = 0; s < S; s++) {
        const int bk = b*K + s*N + n;
        const float rad = sr[bk];
        if (rad < 0.0f) continue;
        const float sx=sc[0*BK+bk], sy=sc[1*BK+bk], sz=sc[2*BK+bk];
        for (int t = 0; t < te; t++)
            min_d[t] = fminf(min_d[t], sphere_sphere_dist(sx, sy, sz, rad,
                sm[t*4], sm[t*4+1], sm[t*4+2], sm[t*4+3]));
    }
    const int bn = b*N + n;
    for (int t = 0; t < te; t++) out[bn*M + mo+ms+t] = min_d[t];
}

static __global__ __launch_bounds__(256, 2)
void wcsr_vs_capsules(
    const float* __restrict__ sc, const float* __restrict__ sr,
    const float* __restrict__ wd, float* __restrict__ out,
    int B, int K, int S, int N, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 7; i += BLOCK_K) sm[i] = wd[ms*7 + i];
    __syncthreads();

    const int b = blockIdx.x, n = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || n >= N) return;

    const int BK = B*K;
    float min_d[TILE_M];
    #pragma unroll
    for (int t = 0; t < TILE_M; t++) min_d[t] = 1e9f;

    for (int s = 0; s < S; s++) {
        const int bk = b*K + s*N + n;
        const float rad = sr[bk];
        if (rad < 0.0f) continue;
        const float sx=sc[0*BK+bk], sy=sc[1*BK+bk], sz=sc[2*BK+bk];
        for (int t = 0; t < te; t++)
            min_d[t] = fminf(min_d[t], sphere_capsule_dist(sx, sy, sz, rad,
                sm[t*7], sm[t*7+1], sm[t*7+2],
                sm[t*7+3], sm[t*7+4], sm[t*7+5], sm[t*7+6]));
    }
    const int bn = b*N + n;
    for (int t = 0; t < te; t++) out[bn*M + mo+ms+t] = min_d[t];
}

static __global__ __launch_bounds__(256, 2)
void wcsr_vs_boxes(
    const float* __restrict__ sc, const float* __restrict__ sr,
    const float* __restrict__ wd, float* __restrict__ out,
    int B, int K, int S, int N, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 15; i += BLOCK_K) sm[i] = wd[ms*15 + i];
    __syncthreads();

    const int b = blockIdx.x, n = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || n >= N) return;

    const int BK = B*K;
    float min_d[TILE_M];
    #pragma unroll
    for (int t = 0; t < TILE_M; t++) min_d[t] = 1e9f;

    for (int s = 0; s < S; s++) {
        const int bk = b*K + s*N + n;
        const float rad = sr[bk];
        if (rad < 0.0f) continue;
        const float sx=sc[0*BK+bk], sy=sc[1*BK+bk], sz=sc[2*BK+bk];
        for (int t = 0; t < te; t++) {
            const float* o = sm + t*15;
            min_d[t] = fminf(min_d[t], sphere_box_dist(sx, sy, sz, rad,
                o[0],o[1],o[2], o[3],o[4],o[5], o[6],o[7],o[8],
                o[9],o[10],o[11], o[12],o[13],o[14]));
        }
    }
    const int bn = b*N + n;
    for (int t = 0; t < te; t++) out[bn*M + mo+ms+t] = min_d[t];
}

static __global__ __launch_bounds__(256, 2)
void wcsr_vs_halfspaces(
    const float* __restrict__ sc, const float* __restrict__ sr,
    const float* __restrict__ wd, float* __restrict__ out,
    int B, int K, int S, int N, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 6; i += BLOCK_K) sm[i] = wd[ms*6 + i];
    __syncthreads();

    const int b = blockIdx.x, n = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || n >= N) return;

    const int BK = B*K;
    float min_d[TILE_M];
    #pragma unroll
    for (int t = 0; t < TILE_M; t++) min_d[t] = 1e9f;

    for (int s = 0; s < S; s++) {
        const int bk = b*K + s*N + n;
        const float rad = sr[bk];
        if (rad < 0.0f) continue;
        const float sx=sc[0*BK+bk], sy=sc[1*BK+bk], sz=sc[2*BK+bk];
        for (int t = 0; t < te; t++)
            min_d[t] = fminf(min_d[t], sphere_halfspace_dist(sx, sy, sz, rad,
                sm[t*6], sm[t*6+1], sm[t*6+2], sm[t*6+3], sm[t*6+4], sm[t*6+5]));
    }
    const int bn = b*N + n;
    for (int t = 0; t < te; t++) out[bn*M + mo+ms+t] = min_d[t];
}

// ── World collision: capsule-robot (output [B, N, M]) ─────────────────────────
//
// Grid:  (B, ceil(N/BLOCK_K), ceil(M_type/TILE_M))
// Caps in SoA [7, B, N]; loads x1,y1,z1,x2,y2,z2,r once then loops over tile.

static __global__ __launch_bounds__(256, 2)
void wcc_vs_spheres(
    const float* __restrict__ caps,  // [7, B, N] SoA
    const float* __restrict__ wd,    // [Ms, 4]
    float*       __restrict__ out,   // [B, N, M_total]
    int B, int N, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 4; i += BLOCK_K) sm[i] = wd[ms*4 + i];
    __syncthreads();

    const int b = blockIdx.x, n = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || n >= N) return;
    const int BN=B*N, bn=b*N+n;
    const float x1=caps[0*BN+bn], y1=caps[1*BN+bn], z1=caps[2*BN+bn];
    const float x2=caps[3*BN+bn], y2=caps[4*BN+bn], z2=caps[5*BN+bn];
    const float cr=caps[6*BN+bn];
    // Sphere–capsule is symmetric: world sphere as "sphere" arg, robot capsule as "capsule"
    for (int t = 0; t < te; t++)
        out[bn*M+mo+ms+t] = sphere_capsule_dist(
            sm[t*4], sm[t*4+1], sm[t*4+2], sm[t*4+3],
            x1, y1, z1, x2, y2, z2, cr);
}

static __global__ __launch_bounds__(256, 2)
void wcc_vs_capsules(
    const float* __restrict__ caps,
    const float* __restrict__ wd,
    float*       __restrict__ out,
    int B, int N, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 7; i += BLOCK_K) sm[i] = wd[ms*7 + i];
    __syncthreads();

    const int b = blockIdx.x, n = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || n >= N) return;
    const int BN=B*N, bn=b*N+n;
    const float x1=caps[0*BN+bn], y1=caps[1*BN+bn], z1=caps[2*BN+bn];
    const float x2=caps[3*BN+bn], y2=caps[4*BN+bn], z2=caps[5*BN+bn];
    const float cr=caps[6*BN+bn];
    for (int t = 0; t < te; t++)
        out[bn*M+mo+ms+t] = capsule_capsule_dist(
            x1, y1, z1, x2, y2, z2, cr,
            sm[t*7], sm[t*7+1], sm[t*7+2],
            sm[t*7+3], sm[t*7+4], sm[t*7+5], sm[t*7+6]);
}

static __global__ __launch_bounds__(256, 2)
void wcc_vs_boxes(
    const float* __restrict__ caps,
    const float* __restrict__ wd,
    float*       __restrict__ out,
    int B, int N, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 15; i += BLOCK_K) sm[i] = wd[ms*15 + i];
    __syncthreads();

    const int b = blockIdx.x, n = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || n >= N) return;
    const int BN=B*N, bn=b*N+n;
    const float x1=caps[0*BN+bn], y1=caps[1*BN+bn], z1=caps[2*BN+bn];
    const float x2=caps[3*BN+bn], y2=caps[4*BN+bn], z2=caps[5*BN+bn];
    const float cr=caps[6*BN+bn];
    for (int t = 0; t < te; t++) {
        const float* o = sm + t*15;
        out[bn*M+mo+ms+t] = capsule_box_dist(
            x1, y1, z1, x2, y2, z2, cr,
            o[0],o[1],o[2], o[3],o[4],o[5], o[6],o[7],o[8],
            o[9],o[10],o[11], o[12],o[13],o[14]);
    }
}

static __global__ __launch_bounds__(256, 2)
void wcc_vs_halfspaces(
    const float* __restrict__ caps,
    const float* __restrict__ wd,
    float*       __restrict__ out,
    int B, int N, int Mt, int M, int mo)
{
    extern __shared__ float sm[];
    const int ms = blockIdx.z * TILE_M, te = min(TILE_M, Mt - ms);
    for (int i = threadIdx.x; i < te * 6; i += BLOCK_K) sm[i] = wd[ms*6 + i];
    __syncthreads();

    const int b = blockIdx.x, n = blockIdx.y * BLOCK_K + threadIdx.x;
    if (b >= B || n >= N) return;
    const int BN=B*N, bn=b*N+n;
    const float x1=caps[0*BN+bn], y1=caps[1*BN+bn], z1=caps[2*BN+bn];
    const float x2=caps[3*BN+bn], y2=caps[4*BN+bn], z2=caps[5*BN+bn];
    const float cr=caps[6*BN+bn];
    for (int t = 0; t < te; t++)
        out[bn*M+mo+ms+t] = capsule_halfspace_dist(
            x1, y1, z1, x2, y2, z2, cr,
            sm[t*6], sm[t*6+1], sm[t*6+2], sm[t*6+3], sm[t*6+4], sm[t*6+5]);
}

// ── Self-collision: sphere-based robot ────────────────────────────────────────

/**
 * self_collision_sphere_kernel
 *
 * One thread per (batch b, active pair p). Iterates over all S×S sphere pairs
 * for the two links (li, lj) and returns the minimum sphere–sphere distance.
 * Padding spheres (radius < 0) are skipped.
 *
 * Inputs:
 *   sphere_centers [B, S, N, 3]  AoS
 *   sphere_radii   [B, S, N]
 *   pair_i, pair_j [P]
 * Output:
 *   out_dist [B, P]
 */
static __global__ __launch_bounds__(256, 2)
void self_collision_sphere_kernel(
    const float* __restrict__ sphere_centers,
    const float* __restrict__ sphere_radii,
    const int*   __restrict__ pair_i,
    const int*   __restrict__ pair_j,
    float*       __restrict__ out_dist,
    int B, int S, int N, int P)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * P) return;

    int p  = idx % P;
    int b  = idx / P;
    int li = pair_i[p];
    int lj = pair_j[p];

    float min_d = 1e9f;
    for (int si = 0; si < S; si++) {
        float ri = sphere_radii[(b*S + si)*N + li];
        if (ri < 0.0f) continue;
        float aix = sphere_centers[((b*S + si)*N + li)*3 + 0];
        float aiy = sphere_centers[((b*S + si)*N + li)*3 + 1];
        float aiz = sphere_centers[((b*S + si)*N + li)*3 + 2];
        for (int sj = 0; sj < S; sj++) {
            float rj = sphere_radii[(b*S + sj)*N + lj];
            if (rj < 0.0f) continue;
            float ajx = sphere_centers[((b*S + sj)*N + lj)*3 + 0];
            float ajy = sphere_centers[((b*S + sj)*N + lj)*3 + 1];
            float ajz = sphere_centers[((b*S + sj)*N + lj)*3 + 2];
            min_d = fminf(min_d, sphere_sphere_dist(aix, aiy, aiz, ri, ajx, ajy, ajz, rj));
        }
    }
    out_dist[b*P + p] = min_d;
}

// ── Self-collision: capsule-based robot ───────────────────────────────────────

/**
 * self_collision_capsule_kernel
 *
 * One thread per (batch b, active pair p).
 *
 * Inputs:
 *   caps [B, N, 7]  AoS — (x1,y1,z1, x2,y2,z2, r) per link
 *   pair_i, pair_j [P]
 * Output:
 *   out_dist [B, P]
 */
static __global__ __launch_bounds__(256, 2)
void self_collision_capsule_kernel(
    const float* __restrict__ caps,
    const int*   __restrict__ pair_i,
    const int*   __restrict__ pair_j,
    float*       __restrict__ out_dist,
    int B, int N, int P)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * P) return;

    int p  = idx % P;
    int b  = idx / P;
    int li = pair_i[p];
    int lj = pair_j[p];

    int ai = (b*N + li) * 7;
    int aj = (b*N + lj) * 7;
    out_dist[b*P + p] = capsule_capsule_dist(
        caps[ai+0], caps[ai+1], caps[ai+2], caps[ai+3], caps[ai+4], caps[ai+5], caps[ai+6],
        caps[aj+0], caps[aj+1], caps[aj+2], caps[aj+3], caps[aj+4], caps[aj+5], caps[aj+6]);
}

// ── XLA FFI handlers ──────────────────────────────────────────────────────────

// Macro: build grid for world collision kernels
#define WORLD_GRID(B_, R_, Mt_) \
    dim3(static_cast<unsigned>(B_), \
         static_cast<unsigned>((R_ + BLOCK_K - 1) / BLOCK_K), \
         static_cast<unsigned>((Mt_ + TILE_M - 1) / TILE_M))

// ── World collision: sphere-based robot, non-reduced → [B, K, M] ─────────────

static ffi::Error CollisionWorldSphereImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> sphere_centers,   // [3, B, K] SoA
    ffi::Buffer<ffi::DataType::F32> sphere_radii,     // [B, K]
    ffi::Buffer<ffi::DataType::F32> world_spheres,    // [Ms, 4]
    ffi::Buffer<ffi::DataType::F32> world_capsules,   // [Mc, 7]
    ffi::Buffer<ffi::DataType::F32> world_boxes,      // [Mb, 15]
    ffi::Buffer<ffi::DataType::F32> world_halfspaces, // [Mh, 6]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out) // [B, K, M]
{
    const int B  = static_cast<int>(sphere_centers.dimensions()[1]);
    const int K  = static_cast<int>(sphere_centers.dimensions()[2]);
    const int Ms = static_cast<int>(world_spheres.dimensions()[0]);
    const int Mc = static_cast<int>(world_capsules.dimensions()[0]);
    const int Mb = static_cast<int>(world_boxes.dimensions()[0]);
    const int Mh = static_cast<int>(world_halfspaces.dimensions()[0]);
    const int M  = Ms + Mc + Mb + Mh;

    if (B > 0 && K > 0 && M > 0) {
        const float* sc = sphere_centers.typed_data();
        const float* sr = sphere_radii.typed_data();
        float* od = out->typed_data();

        if (Ms > 0) wcs_vs_spheres<<<WORLD_GRID(B,K,Ms), BLOCK_K, TILE_M*4*sizeof(float), stream>>>(
            sc, sr, world_spheres.typed_data(), od, B, K, Ms, M, 0);
        if (Mc > 0) wcs_vs_capsules<<<WORLD_GRID(B,K,Mc), BLOCK_K, TILE_M*7*sizeof(float), stream>>>(
            sc, sr, world_capsules.typed_data(), od, B, K, Mc, M, Ms);
        if (Mb > 0) wcs_vs_boxes<<<WORLD_GRID(B,K,Mb), BLOCK_K, TILE_M*15*sizeof(float), stream>>>(
            sc, sr, world_boxes.typed_data(), od, B, K, Mb, M, Ms+Mc);
        if (Mh > 0) wcs_vs_halfspaces<<<WORLD_GRID(B,K,Mh), BLOCK_K, TILE_M*6*sizeof(float), stream>>>(
            sc, sr, world_halfspaces.typed_data(), od, B, K, Mh, M, Ms+Mc+Mb);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CollisionWorldSphereFfi, CollisionWorldSphereImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // sphere_centers [3, B, K]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // sphere_radii   [B, K]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out [B, K, M]

// ── World collision: sphere-based robot, fused S-reduction → [B, N, M] ────────

static ffi::Error CollisionWorldSphereReducedImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> sphere_centers,   // [3, B, K] SoA, K = S*N
    ffi::Buffer<ffi::DataType::F32> sphere_radii,     // [B, K]
    ffi::Buffer<ffi::DataType::F32> world_spheres,    // [Ms, 4]
    ffi::Buffer<ffi::DataType::F32> world_capsules,   // [Mc, 7]
    ffi::Buffer<ffi::DataType::F32> world_boxes,      // [Mb, 15]
    ffi::Buffer<ffi::DataType::F32> world_halfspaces, // [Mh, 6]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out, // [B, N, M]
    int64_t n)                                        // attribute: N = num_links
{
    const int B  = static_cast<int>(sphere_centers.dimensions()[1]);
    const int K  = static_cast<int>(sphere_centers.dimensions()[2]);
    const int N  = static_cast<int>(n);
    const int S  = (N > 0) ? K / N : 0;
    const int Ms = static_cast<int>(world_spheres.dimensions()[0]);
    const int Mc = static_cast<int>(world_capsules.dimensions()[0]);
    const int Mb = static_cast<int>(world_boxes.dimensions()[0]);
    const int Mh = static_cast<int>(world_halfspaces.dimensions()[0]);
    const int M  = Ms + Mc + Mb + Mh;

    if (B > 0 && N > 0 && M > 0) {
        const float* sc = sphere_centers.typed_data();
        const float* sr = sphere_radii.typed_data();
        float* od = out->typed_data();

        if (Ms > 0) wcsr_vs_spheres<<<WORLD_GRID(B,N,Ms), BLOCK_K, TILE_M*4*sizeof(float), stream>>>(
            sc, sr, world_spheres.typed_data(), od, B, K, S, N, Ms, M, 0);
        if (Mc > 0) wcsr_vs_capsules<<<WORLD_GRID(B,N,Mc), BLOCK_K, TILE_M*7*sizeof(float), stream>>>(
            sc, sr, world_capsules.typed_data(), od, B, K, S, N, Mc, M, Ms);
        if (Mb > 0) wcsr_vs_boxes<<<WORLD_GRID(B,N,Mb), BLOCK_K, TILE_M*15*sizeof(float), stream>>>(
            sc, sr, world_boxes.typed_data(), od, B, K, S, N, Mb, M, Ms+Mc);
        if (Mh > 0) wcsr_vs_halfspaces<<<WORLD_GRID(B,N,Mh), BLOCK_K, TILE_M*6*sizeof(float), stream>>>(
            sc, sr, world_halfspaces.typed_data(), od, B, K, S, N, Mh, M, Ms+Mc+Mb);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CollisionWorldSphereReducedFfi, CollisionWorldSphereReducedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // sphere_centers [3, B, K]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // sphere_radii   [B, K]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // out [B, N, M]
        .Attr<int64_t>("n"));                     // num_links N

// ── World collision: capsule-based robot → [B, N, M] ─────────────────────────

static ffi::Error CollisionWorldCapsuleImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> caps,             // [7, B, N] SoA
    ffi::Buffer<ffi::DataType::F32> world_spheres,    // [Ms, 4]
    ffi::Buffer<ffi::DataType::F32> world_capsules,   // [Mc, 7]
    ffi::Buffer<ffi::DataType::F32> world_boxes,      // [Mb, 15]
    ffi::Buffer<ffi::DataType::F32> world_halfspaces, // [Mh, 6]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out) // [B, N, M]
{
    const int B  = static_cast<int>(caps.dimensions()[1]);
    const int N  = static_cast<int>(caps.dimensions()[2]);
    const int Ms = static_cast<int>(world_spheres.dimensions()[0]);
    const int Mc = static_cast<int>(world_capsules.dimensions()[0]);
    const int Mb = static_cast<int>(world_boxes.dimensions()[0]);
    const int Mh = static_cast<int>(world_halfspaces.dimensions()[0]);
    const int M  = Ms + Mc + Mb + Mh;

    if (B > 0 && N > 0 && M > 0) {
        const float* cp = caps.typed_data();
        float* od = out->typed_data();

        if (Ms > 0) wcc_vs_spheres<<<WORLD_GRID(B,N,Ms), BLOCK_K, TILE_M*4*sizeof(float), stream>>>(
            cp, world_spheres.typed_data(), od, B, N, Ms, M, 0);
        if (Mc > 0) wcc_vs_capsules<<<WORLD_GRID(B,N,Mc), BLOCK_K, TILE_M*7*sizeof(float), stream>>>(
            cp, world_capsules.typed_data(), od, B, N, Mc, M, Ms);
        if (Mb > 0) wcc_vs_boxes<<<WORLD_GRID(B,N,Mb), BLOCK_K, TILE_M*15*sizeof(float), stream>>>(
            cp, world_boxes.typed_data(), od, B, N, Mb, M, Ms+Mc);
        if (Mh > 0) wcc_vs_halfspaces<<<WORLD_GRID(B,N,Mh), BLOCK_K, TILE_M*6*sizeof(float), stream>>>(
            cp, world_halfspaces.typed_data(), od, B, N, Mh, M, Ms+Mc+Mb);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CollisionWorldCapsuleFfi, CollisionWorldCapsuleImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // caps [7, B, N]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // world_halfspaces
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out [B, N, M]

// ── Self-collision: sphere-based robot ───────────────────────────────────────

static ffi::Error CollisionSelfSphereImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> sphere_centers,   // [B, S, N, 3]  AoS
    ffi::Buffer<ffi::DataType::F32> sphere_radii,     // [B, S, N]
    ffi::Buffer<ffi::DataType::S32> pair_i,           // [P]
    ffi::Buffer<ffi::DataType::S32> pair_j,           // [P]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out) // [B, P]
{
    const int B = static_cast<int>(sphere_centers.dimensions()[0]);
    const int S = static_cast<int>(sphere_centers.dimensions()[1]);
    const int N = static_cast<int>(sphere_centers.dimensions()[2]);
    const int P = static_cast<int>(pair_i.dimensions()[0]);
    const int total = B * P;

    if (total > 0) {
        const int blocks = (total + 255) / 256;
        self_collision_sphere_kernel<<<blocks, 256, 0, stream>>>(
            sphere_centers.typed_data(),
            sphere_radii.typed_data(),
            pair_i.typed_data(),
            pair_j.typed_data(),
            out->typed_data(),
            B, S, N, P);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CollisionSelfSphereFfi, CollisionSelfSphereImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // sphere_centers [B, S, N, 3]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // sphere_radii   [B, S, N]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // pair_j
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out [B, P]

// ── Self-collision: capsule-based robot ──────────────────────────────────────

static ffi::Error CollisionSelfCapsuleImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> caps,             // [B, N, 7]  AoS
    ffi::Buffer<ffi::DataType::S32> pair_i,           // [P]
    ffi::Buffer<ffi::DataType::S32> pair_j,           // [P]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out) // [B, P]
{
    const int B = static_cast<int>(caps.dimensions()[0]);
    const int N = static_cast<int>(caps.dimensions()[1]);
    const int P = static_cast<int>(pair_i.dimensions()[0]);
    const int total = B * P;

    if (total > 0) {
        const int blocks = (total + 255) / 256;
        self_collision_capsule_kernel<<<blocks, 256, 0, stream>>>(
            caps.typed_data(),
            pair_i.typed_data(),
            pair_j.typed_data(),
            out->typed_data(),
            B, N, P);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CollisionSelfCapsuleFfi, CollisionSelfCapsuleImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // caps [B, N, 7]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // pair_i
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // pair_j
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out [B, P]
