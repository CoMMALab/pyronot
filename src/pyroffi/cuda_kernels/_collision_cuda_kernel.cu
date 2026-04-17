/**
 * Batched collision-distance CUDA kernels for pyroffi, with XLA/JAX FFI bindings.
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
 * Distance convention (matches pyroffi JAX backend):
 *   positive → separated
 *   negative → penetration
 *
 * Build: bash src/pyroffi/cuda_kernels/build_collision_cuda.sh
 */

#include "xla/ffi/api/ffi.h"
#include "_collision_cuda_helpers.cuh"

namespace ffi = xla::ffi;

// ── Grid / tile constants ─────────────────────────────────────────────────────

/// Threads per block along the robot (K or N) dimension.
constexpr int BLOCK_K = 256;

/// World obstacles loaded into shared memory per tile.
constexpr int TILE_M  = 16;

// Geometry primitives are shared across collision-enabled kernels.

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

    struct GraphCache {
        cudaGraphExec_t exec = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphNode_t nodes[4] = {nullptr, nullptr, nullptr, nullptr};
        size_t node_count = 0;
        int B = -1, K = -1, Ms = -1, Mc = -1, Mb = -1, Mh = -1;
        bool shape_matches(int b, int k, int ms, int mc, int mb, int mh) const noexcept {
            return b == B && k == K && ms == Ms && mc == Mc && mb == Mb && mh == Mh;
        }
        void invalidate() noexcept {
            if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
            node_count = 0;
            B = K = Ms = Mc = Mb = Mh = -1;
        }
    };
    static GraphCache cache;

    if (B > 0 && K > 0 && M > 0) {
        const float* sc = sphere_centers.typed_data();
        const float* sr = sphere_radii.typed_data();
        float* od = out->typed_data();

        if (!cache.shape_matches(B, K, Ms, Mc, Mb, Mh)) {
            cache.invalidate();
            cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess)
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

            if (Ms > 0) wcs_vs_spheres<<<WORLD_GRID(B,K,Ms), BLOCK_K, TILE_M*4*sizeof(float), stream>>>(
                sc, sr, world_spheres.typed_data(), od, B, K, Ms, M, 0);
            if (Mc > 0) wcs_vs_capsules<<<WORLD_GRID(B,K,Mc), BLOCK_K, TILE_M*7*sizeof(float), stream>>>(
                sc, sr, world_capsules.typed_data(), od, B, K, Mc, M, Ms);
            if (Mb > 0) wcs_vs_boxes<<<WORLD_GRID(B,K,Mb), BLOCK_K, TILE_M*15*sizeof(float), stream>>>(
                sc, sr, world_boxes.typed_data(), od, B, K, Mb, M, Ms+Mc);
            if (Mh > 0) wcs_vs_halfspaces<<<WORLD_GRID(B,K,Mh), BLOCK_K, TILE_M*6*sizeof(float), stream>>>(
                sc, sr, world_halfspaces.typed_data(), od, B, K, Mh, M, Ms+Mc+Mb);

            e = cudaGetLastError();
            if (e != cudaSuccess) {
                cudaStreamEndCapture(stream, nullptr);
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            e = cudaStreamEndCapture(stream, &cache.graph);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            size_t n_nodes = 0;
            e = cudaGraphGetNodes(cache.graph, nullptr, &n_nodes);
            if (e != cudaSuccess || n_nodes == 0 || n_nodes > 4) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal,
                    (e != cudaSuccess) ? cudaGetErrorString(e) : "Unexpected node count in collision world sphere graph.");
            }
            cache.node_count = n_nodes;
            e = cudaGraphGetNodes(cache.graph, cache.nodes, &cache.node_count);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            e = cudaGraphInstantiate(&cache.exec, cache.graph, nullptr, nullptr, 0);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            cache.B = B; cache.K = K; cache.Ms = Ms; cache.Mc = Mc; cache.Mb = Mb; cache.Mh = Mh;
        } else {
            int node_idx = 0;
            if (Ms > 0) {
                void* wd = const_cast<float*>(world_spheres.typed_data());
                int kB=B, kK=K, kMt=Ms, kM=M, kMo=0;
                void* args[] = {&sc, &sr, &wd, &od, &kB, &kK, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcs_vs_spheres);
                kp.gridDim = WORLD_GRID(B,K,Ms);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 4 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mc > 0) {
                void* wd = const_cast<float*>(world_capsules.typed_data());
                int kB=B, kK=K, kMt=Mc, kM=M, kMo=Ms;
                void* args[] = {&sc, &sr, &wd, &od, &kB, &kK, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcs_vs_capsules);
                kp.gridDim = WORLD_GRID(B,K,Mc);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 7 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mb > 0) {
                void* wd = const_cast<float*>(world_boxes.typed_data());
                int kB=B, kK=K, kMt=Mb, kM=M, kMo=Ms+Mc;
                void* args[] = {&sc, &sr, &wd, &od, &kB, &kK, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcs_vs_boxes);
                kp.gridDim = WORLD_GRID(B,K,Mb);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 15 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mh > 0) {
                void* wd = const_cast<float*>(world_halfspaces.typed_data());
                int kB=B, kK=K, kMt=Mh, kM=M, kMo=Ms+Mc+Mb;
                void* args[] = {&sc, &sr, &wd, &od, &kB, &kK, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcs_vs_halfspaces);
                kp.gridDim = WORLD_GRID(B,K,Mh);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 6 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
        }

        const cudaError_t launch_err = cudaGraphLaunch(cache.exec, stream);
        if (launch_err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(launch_err));
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

    struct GraphCache {
        cudaGraphExec_t exec = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphNode_t nodes[4] = {nullptr, nullptr, nullptr, nullptr};
        size_t node_count = 0;
        int B = -1, K = -1, N = -1, Ms = -1, Mc = -1, Mb = -1, Mh = -1;
        bool shape_matches(int b, int k, int nn, int ms, int mc, int mb, int mh) const noexcept {
            return b == B && k == K && nn == N && ms == Ms && mc == Mc && mb == Mb && mh == Mh;
        }
        void invalidate() noexcept {
            if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
            node_count = 0;
            B = K = N = Ms = Mc = Mb = Mh = -1;
        }
    };
    static GraphCache cache;

    if (B > 0 && N > 0 && M > 0) {
        const float* sc = sphere_centers.typed_data();
        const float* sr = sphere_radii.typed_data();
        float* od = out->typed_data();

        if (!cache.shape_matches(B, K, N, Ms, Mc, Mb, Mh)) {
            cache.invalidate();
            cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess)
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

            if (Ms > 0) wcsr_vs_spheres<<<WORLD_GRID(B,N,Ms), BLOCK_K, TILE_M*4*sizeof(float), stream>>>(
                sc, sr, world_spheres.typed_data(), od, B, K, S, N, Ms, M, 0);
            if (Mc > 0) wcsr_vs_capsules<<<WORLD_GRID(B,N,Mc), BLOCK_K, TILE_M*7*sizeof(float), stream>>>(
                sc, sr, world_capsules.typed_data(), od, B, K, S, N, Mc, M, Ms);
            if (Mb > 0) wcsr_vs_boxes<<<WORLD_GRID(B,N,Mb), BLOCK_K, TILE_M*15*sizeof(float), stream>>>(
                sc, sr, world_boxes.typed_data(), od, B, K, S, N, Mb, M, Ms+Mc);
            if (Mh > 0) wcsr_vs_halfspaces<<<WORLD_GRID(B,N,Mh), BLOCK_K, TILE_M*6*sizeof(float), stream>>>(
                sc, sr, world_halfspaces.typed_data(), od, B, K, S, N, Mh, M, Ms+Mc+Mb);

            e = cudaGetLastError();
            if (e != cudaSuccess) {
                cudaStreamEndCapture(stream, nullptr);
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            e = cudaStreamEndCapture(stream, &cache.graph);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            size_t n_nodes = 0;
            e = cudaGraphGetNodes(cache.graph, nullptr, &n_nodes);
            if (e != cudaSuccess || n_nodes == 0 || n_nodes > 4) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal,
                    (e != cudaSuccess) ? cudaGetErrorString(e) : "Unexpected node count in reduced collision graph.");
            }
            cache.node_count = n_nodes;
            e = cudaGraphGetNodes(cache.graph, cache.nodes, &cache.node_count);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            e = cudaGraphInstantiate(&cache.exec, cache.graph, nullptr, nullptr, 0);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            cache.B = B; cache.K = K; cache.N = N; cache.Ms = Ms; cache.Mc = Mc; cache.Mb = Mb; cache.Mh = Mh;
        } else {
            int node_idx = 0;
            if (Ms > 0) {
                void* wd = const_cast<float*>(world_spheres.typed_data());
                int kB=B, kK=K, kS=S, kN=N, kMt=Ms, kM=M, kMo=0;
                void* args[] = {&sc, &sr, &wd, &od, &kB, &kK, &kS, &kN, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcsr_vs_spheres);
                kp.gridDim = WORLD_GRID(B,N,Ms);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 4 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mc > 0) {
                void* wd = const_cast<float*>(world_capsules.typed_data());
                int kB=B, kK=K, kS=S, kN=N, kMt=Mc, kM=M, kMo=Ms;
                void* args[] = {&sc, &sr, &wd, &od, &kB, &kK, &kS, &kN, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcsr_vs_capsules);
                kp.gridDim = WORLD_GRID(B,N,Mc);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 7 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mb > 0) {
                void* wd = const_cast<float*>(world_boxes.typed_data());
                int kB=B, kK=K, kS=S, kN=N, kMt=Mb, kM=M, kMo=Ms+Mc;
                void* args[] = {&sc, &sr, &wd, &od, &kB, &kK, &kS, &kN, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcsr_vs_boxes);
                kp.gridDim = WORLD_GRID(B,N,Mb);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 15 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mh > 0) {
                void* wd = const_cast<float*>(world_halfspaces.typed_data());
                int kB=B, kK=K, kS=S, kN=N, kMt=Mh, kM=M, kMo=Ms+Mc+Mb;
                void* args[] = {&sc, &sr, &wd, &od, &kB, &kK, &kS, &kN, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcsr_vs_halfspaces);
                kp.gridDim = WORLD_GRID(B,N,Mh);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 6 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
        }

        const cudaError_t launch_err = cudaGraphLaunch(cache.exec, stream);
        if (launch_err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(launch_err));
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

    struct GraphCache {
        cudaGraphExec_t exec = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphNode_t nodes[4] = {nullptr, nullptr, nullptr, nullptr};
        size_t node_count = 0;
        int B = -1, N = -1, Ms = -1, Mc = -1, Mb = -1, Mh = -1;
        bool shape_matches(int b, int n, int ms, int mc, int mb, int mh) const noexcept {
            return b == B && n == N && ms == Ms && mc == Mc && mb == Mb && mh == Mh;
        }
        void invalidate() noexcept {
            if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
            node_count = 0;
            B = N = Ms = Mc = Mb = Mh = -1;
        }
    };
    static GraphCache cache;

    if (B > 0 && N > 0 && M > 0) {
        const float* cp = caps.typed_data();
        float* od = out->typed_data();

        if (!cache.shape_matches(B, N, Ms, Mc, Mb, Mh)) {
            cache.invalidate();
            cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess)
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

            if (Ms > 0) wcc_vs_spheres<<<WORLD_GRID(B,N,Ms), BLOCK_K, TILE_M*4*sizeof(float), stream>>>(
                cp, world_spheres.typed_data(), od, B, N, Ms, M, 0);
            if (Mc > 0) wcc_vs_capsules<<<WORLD_GRID(B,N,Mc), BLOCK_K, TILE_M*7*sizeof(float), stream>>>(
                cp, world_capsules.typed_data(), od, B, N, Mc, M, Ms);
            if (Mb > 0) wcc_vs_boxes<<<WORLD_GRID(B,N,Mb), BLOCK_K, TILE_M*15*sizeof(float), stream>>>(
                cp, world_boxes.typed_data(), od, B, N, Mb, M, Ms+Mc);
            if (Mh > 0) wcc_vs_halfspaces<<<WORLD_GRID(B,N,Mh), BLOCK_K, TILE_M*6*sizeof(float), stream>>>(
                cp, world_halfspaces.typed_data(), od, B, N, Mh, M, Ms+Mc+Mb);

            e = cudaGetLastError();
            if (e != cudaSuccess) {
                cudaStreamEndCapture(stream, nullptr);
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            e = cudaStreamEndCapture(stream, &cache.graph);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            size_t n_nodes = 0;
            e = cudaGraphGetNodes(cache.graph, nullptr, &n_nodes);
            if (e != cudaSuccess || n_nodes == 0 || n_nodes > 4) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal,
                    (e != cudaSuccess) ? cudaGetErrorString(e) : "Unexpected node count in capsule collision graph.");
            }
            cache.node_count = n_nodes;
            e = cudaGraphGetNodes(cache.graph, cache.nodes, &cache.node_count);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            e = cudaGraphInstantiate(&cache.exec, cache.graph, nullptr, nullptr, 0);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            cache.B = B; cache.N = N; cache.Ms = Ms; cache.Mc = Mc; cache.Mb = Mb; cache.Mh = Mh;
        } else {
            int node_idx = 0;
            if (Ms > 0) {
                void* wd = const_cast<float*>(world_spheres.typed_data());
                int kB=B, kN=N, kMt=Ms, kM=M, kMo=0;
                void* args[] = {&cp, &wd, &od, &kB, &kN, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcc_vs_spheres);
                kp.gridDim = WORLD_GRID(B,N,Ms);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 4 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mc > 0) {
                void* wd = const_cast<float*>(world_capsules.typed_data());
                int kB=B, kN=N, kMt=Mc, kM=M, kMo=Ms;
                void* args[] = {&cp, &wd, &od, &kB, &kN, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcc_vs_capsules);
                kp.gridDim = WORLD_GRID(B,N,Mc);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 7 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mb > 0) {
                void* wd = const_cast<float*>(world_boxes.typed_data());
                int kB=B, kN=N, kMt=Mb, kM=M, kMo=Ms+Mc;
                void* args[] = {&cp, &wd, &od, &kB, &kN, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcc_vs_boxes);
                kp.gridDim = WORLD_GRID(B,N,Mb);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 15 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
            if (Mh > 0) {
                void* wd = const_cast<float*>(world_halfspaces.typed_data());
                int kB=B, kN=N, kMt=Mh, kM=M, kMo=Ms+Mc+Mb;
                void* args[] = {&cp, &wd, &od, &kB, &kN, &kMt, &kM, &kMo};
                cudaKernelNodeParams kp = {};
                kp.func = reinterpret_cast<void*>(wcc_vs_halfspaces);
                kp.gridDim = WORLD_GRID(B,N,Mh);
                kp.blockDim = dim3(BLOCK_K,1,1);
                kp.sharedMemBytes = TILE_M * 6 * sizeof(float);
                kp.kernelParams = args;
                cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.nodes[node_idx++], &kp);
                if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }
        }

        const cudaError_t launch_err = cudaGraphLaunch(cache.exec, stream);
        if (launch_err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(launch_err));
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

    struct GraphCache {
        cudaGraphExec_t exec = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphNode_t node = nullptr;
        int B = -1, S = -1, N = -1, P = -1;
        bool shape_matches(int b, int s, int n, int p) const noexcept {
            return b == B && s == S && n == N && p == P;
        }
        void invalidate() noexcept {
            if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
            node = nullptr;
            B = S = N = P = -1;
        }
    };
    static GraphCache cache;

    if (total > 0) {
        const int blocks = (total + 255) / 256;
        const float* centers = sphere_centers.typed_data();
        const float* radii = sphere_radii.typed_data();
        const int* pi = pair_i.typed_data();
        const int* pj = pair_j.typed_data();
        float* od = out->typed_data();

        if (!cache.shape_matches(B, S, N, P)) {
            cache.invalidate();
            cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess)
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

            self_collision_sphere_kernel<<<blocks, 256, 0, stream>>>(
                centers, radii, pi, pj, od, B, S, N, P);

            e = cudaGetLastError();
            if (e != cudaSuccess) {
                cudaStreamEndCapture(stream, nullptr);
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            e = cudaStreamEndCapture(stream, &cache.graph);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            size_t n_nodes = 1;
            e = cudaGraphGetNodes(cache.graph, &cache.node, &n_nodes);
            if (e != cudaSuccess || n_nodes == 0) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal,
                                  (e != cudaSuccess) ? cudaGetErrorString(e) : "Self-sphere graph missing node.");
            }

            e = cudaGraphInstantiate(&cache.exec, cache.graph, nullptr, nullptr, 0);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            cache.B = B; cache.S = S; cache.N = N; cache.P = P;
        } else {
            int kB = B, kS = S, kN = N, kP = P;
            void* args[] = {&centers, &radii, &pi, &pj, &od, &kB, &kS, &kN, &kP};
            cudaKernelNodeParams kp = {};
            kp.func = reinterpret_cast<void*>(self_collision_sphere_kernel);
            kp.gridDim = dim3(static_cast<unsigned>(blocks), 1u, 1u);
            kp.blockDim = dim3(256u, 1u, 1u);
            kp.sharedMemBytes = 0;
            kp.kernelParams = args;
            cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.node, &kp);
            if (e != cudaSuccess)
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
        }

        cudaError_t launch_err = cudaGraphLaunch(cache.exec, stream);
        if (launch_err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(launch_err));
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

    struct GraphCache {
        cudaGraphExec_t exec = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphNode_t node = nullptr;
        int B = -1, N = -1, P = -1;
        bool shape_matches(int b, int n, int p) const noexcept {
            return b == B && n == N && p == P;
        }
        void invalidate() noexcept {
            if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
            node = nullptr;
            B = N = P = -1;
        }
    };
    static GraphCache cache;

    if (total > 0) {
        const int blocks = (total + 255) / 256;
        const float* cp = caps.typed_data();
        const int* pi = pair_i.typed_data();
        const int* pj = pair_j.typed_data();
        float* od = out->typed_data();

        if (!cache.shape_matches(B, N, P)) {
            cache.invalidate();
            cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess)
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));

            self_collision_capsule_kernel<<<blocks, 256, 0, stream>>>(
                cp, pi, pj, od, B, N, P);

            e = cudaGetLastError();
            if (e != cudaSuccess) {
                cudaStreamEndCapture(stream, nullptr);
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            e = cudaStreamEndCapture(stream, &cache.graph);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            size_t n_nodes = 1;
            e = cudaGraphGetNodes(cache.graph, &cache.node, &n_nodes);
            if (e != cudaSuccess || n_nodes == 0) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal,
                                  (e != cudaSuccess) ? cudaGetErrorString(e) : "Self-capsule graph missing node.");
            }

            e = cudaGraphInstantiate(&cache.exec, cache.graph, nullptr, nullptr, 0);
            if (e != cudaSuccess) {
                cache.invalidate();
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            }

            cache.B = B; cache.N = N; cache.P = P;
        } else {
            int kB = B, kN = N, kP = P;
            void* args[] = {&cp, &pi, &pj, &od, &kB, &kN, &kP};
            cudaKernelNodeParams kp = {};
            kp.func = reinterpret_cast<void*>(self_collision_capsule_kernel);
            kp.gridDim = dim3(static_cast<unsigned>(blocks), 1u, 1u);
            kp.blockDim = dim3(256u, 1u, 1u);
            kp.sharedMemBytes = 0;
            kp.kernelParams = args;
            cudaError_t e = cudaGraphExecKernelNodeSetParams(cache.exec, cache.node, &kp);
            if (e != cudaSuccess)
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
        }

        cudaError_t launch_err = cudaGraphLaunch(cache.exec, stream);
        if (launch_err != cudaSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(launch_err));
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
