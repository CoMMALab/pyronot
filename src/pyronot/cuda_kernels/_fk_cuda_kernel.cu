/**
 * Forward kinematics CUDA kernel with XLA FFI binding.
 *
 * Computes SE(3) world-frame transforms for every joint given the actuated
 * joint configuration.  Mirrors the JAX implementation in _robot.py exactly:
 *
 *   1. Expand actuated config to full joint config (handles fixed / mimic joints).
 *   2. Compute delta transform  delta_T[j] = SE3.exp(twists[j] * q_full[j]).
 *   3. Combine with constant parent offset: T_pc[j] = parent_tf[j] @ delta_T[j].
 *   4. Walk joints in topological order to accumulate world transforms.
 *
 * SE(3) math and the single-thread FK device function live in
 * _fk_cuda_helpers.cuh so that _ik_cuda_kernel.cu can reuse them.
 *
 * Build with:  bash src/pyronot/cuda_kernels/build_fk_cuda.sh
 */

#include "_fk_cuda_helpers.cuh"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Maximum number of joints supported by the FK shared-memory cache.
// Increase if your robot has more joints.
#define FK_MAX_JOINTS 64

// Robot model constants cached in device constant memory for FK launches.
__device__ __constant__ float c_twists[FK_MAX_JOINTS * 6];
__device__ __constant__ float c_parent_tf[FK_MAX_JOINTS * 7];
__device__ __constant__ int   c_parent_idx[FK_MAX_JOINTS];
__device__ __constant__ int   c_act_idx[FK_MAX_JOINTS];
__device__ __constant__ float c_mimic_mul[FK_MAX_JOINTS];
__device__ __constant__ float c_mimic_off[FK_MAX_JOINTS];
__device__ __constant__ int   c_mimic_act_idx[FK_MAX_JOINTS];
__device__ __constant__ int   c_fk_level_starts[FK_MAX_JOINTS + 1];
__device__ __constant__ int   c_fk_level_joints[FK_MAX_JOINTS];

__device__ __forceinline__
void fk_eval_joint(
    int j,
    const float* __restrict__ cfg_b,
    float*       __restrict__ out_b)
{
    const int m_idx = c_mimic_act_idx[j];
    const int a_idx = c_act_idx[j];
    const int src = (m_idx != -1) ? m_idx : a_idx;
    const float q_ref = (src == -1) ? 0.0f : cfg_b[src];
    const float q_j = q_ref * c_mimic_mul[j] + c_mimic_off[j];

    float T_pc[7];
    if (src == -1 && c_mimic_off[j] == 0.0f) {
        #pragma unroll
        for (int k = 0; k < 7; ++k) T_pc[k] = c_parent_tf[j * 7 + k];
    } else {
        float tangent[6];
        #pragma unroll
        for (int k = 0; k < 6; ++k)
            tangent[k] = c_twists[j * 6 + k] * q_j;

        float delta_T[7];
        se3_exp(tangent, delta_T);
        se3_compose(c_parent_tf + j * 7, delta_T, T_pc);
    }

    float* dst = out_b + j * 7;
    const int p = c_parent_idx[j];
    if (p == -1) {
        #pragma unroll
        for (int k = 0; k < 7; ++k) dst[k] = T_pc[k];
    } else {
        se3_compose(out_b + p * 7, T_pc, dst);
    }
}

// ---------------------------------------------------------------------------
// FK kernel
// ---------------------------------------------------------------------------

/**
 * Warp-packed FK kernel:
 *   - Each warp handles ITEMS_PER_WARP batch items simultaneously.
 *   - Warp lanes are partitioned into sub-groups of LANES_PER_ITEM lanes.
 *   - Within each sub-group, lanes process joints in a level in parallel.
 *   - Dynamic shared memory is sized to actual n_joints (not FK_MAX_JOINTS).
 *
 * Template parameters:
 *   THREADS          — threads per block (must be multiple of 32)
 *   ITEMS_PER_WARP   — batch items packed into one warp (1, 2, 4, 8, or 16)
 */
template <int THREADS, int ITEMS_PER_WARP>
__global__
void fk_kernel_warp(const float* __restrict__ cfg,
                    float*       __restrict__ out,
                    int batch, int n_joints, int n_act, int n_levels)
{
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of warp size");
    constexpr int WARP = 32;
    constexpr int LANES_PER_ITEM = WARP / ITEMS_PER_WARP;
    constexpr int WARPS_PER_BLOCK = THREADS / WARP;
    constexpr int ITEMS_PER_BLOCK = WARPS_PER_BLOCK * ITEMS_PER_WARP;

    const int warp_id   = threadIdx.x / WARP;
    const int lane_id   = threadIdx.x & (WARP - 1);
    const int sub_group = lane_id / LANES_PER_ITEM;
    const int sub_lane  = lane_id % LANES_PER_ITEM;

    const int b = blockIdx.x * ITEMS_PER_BLOCK + warp_id * ITEMS_PER_WARP + sub_group;
    if (b >= batch) return;

    // Dynamic shared memory, sized to actual n_joints at launch.
    extern __shared__ float s_world[];
    const int item_stride = n_joints * 7;
    float* my_world = s_world + (warp_id * ITEMS_PER_WARP + sub_group) * item_stride;

    const float* cfg_b = cfg + (long long)b * n_act;
    float* out_b = out + (long long)b * n_joints * 7;

    // Sub-group mask: only the lanes belonging to our batch item.
    const unsigned sub_mask = ((1u << LANES_PER_ITEM) - 1u)
                              << (sub_group * LANES_PER_ITEM);

    for (int lvl = 0; lvl < n_levels; ++lvl) {
        const int begin = c_fk_level_starts[lvl];
        const int end   = c_fk_level_starts[lvl + 1];
        for (int idx = begin + sub_lane; idx < end; idx += LANES_PER_ITEM) {
            const int j = c_fk_level_joints[idx];
            fk_eval_joint(j, cfg_b, my_world);
        }
        __syncwarp(sub_mask);
    }

    for (int i = sub_lane; i < item_stride; i += LANES_PER_ITEM)
        out_b[i] = my_world[i];
}

template <int THREADS, int ITEMS_PER_WARP>
static inline ffi::Error launch_fk_kernel_warp(
    cudaStream_t stream,
    const float* cfg,
    float* out,
    int batch,
    int n_joints,
    int n_act,
    int n_levels)
{
    constexpr int WARP = 32;
    constexpr int WARPS_PER_BLOCK = THREADS / WARP;
    constexpr int ITEMS_PER_BLOCK = WARPS_PER_BLOCK * ITEMS_PER_WARP;
    int blocks = (batch + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    if (blocks < 1) blocks = 1;

    // Dynamic shared memory: each block processes ITEMS_PER_BLOCK batch items,
    // each needing n_joints * 7 floats of workspace.
    const size_t smem = ITEMS_PER_BLOCK * n_joints * 7 * sizeof(float);

    fk_kernel_warp<THREADS, ITEMS_PER_WARP><<<blocks, THREADS, smem, stream>>>(
        cfg,
        out,
        batch,
        n_joints,
        n_act,
        n_levels);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));

    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

static ffi::Error FkCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> cfg,
    ffi::Buffer<ffi::DataType::F32> twists,
    ffi::Buffer<ffi::DataType::F32> parent_tf,
    ffi::Buffer<ffi::DataType::S32> parent_idx,
    ffi::Buffer<ffi::DataType::S32> act_idx,
    ffi::Buffer<ffi::DataType::F32> mimic_mul,
    ffi::Buffer<ffi::DataType::F32> mimic_off,
    ffi::Buffer<ffi::DataType::S32> mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> topo_inv,
    ffi::Buffer<ffi::DataType::S32> fk_level_starts,
    ffi::Buffer<ffi::DataType::S32> fk_level_joints,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out)
{
    struct ModelCache {
        const void* twists = nullptr;
        const void* parent_tf = nullptr;
        const void* parent_idx = nullptr;
        const void* act_idx = nullptr;
        const void* mimic_mul = nullptr;
        const void* mimic_off = nullptr;
        const void* mimic_act_idx = nullptr;
        const void* fk_level_starts = nullptr;
        const void* fk_level_joints = nullptr;
        int n_joints = -1;
        int n_levels = -1;
        int max_level_width = -1;
        bool valid = false;
    };
    static ModelCache cache;

    const int batch    = static_cast<int>(cfg.dimensions()[0]);
    const int n_act    = static_cast<int>(cfg.dimensions()[1]);
    const int n_joints = static_cast<int>(twists.dimensions()[0]);
    const int n_levels = static_cast<int>(fk_level_starts.dimensions()[0] - 1);

    if (batch <= 0 || n_joints <= 0)
        return ffi::Error::Success();
    if (n_joints > FK_MAX_JOINTS || n_levels > FK_MAX_JOINTS) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "FK kernel supports up to FK_MAX_JOINTS joints/levels; increase FK_MAX_JOINTS and rebuild."
        );
    }

    auto copy_const = [&](const void* src, const void* symbol, size_t bytes) -> bool {
        const cudaError_t e = cudaMemcpyToSymbolAsync(
            symbol, src, bytes, 0, cudaMemcpyDeviceToDevice, stream);
        return e == cudaSuccess;
    };

    const void* twists_ptr = twists.typed_data();
    const void* parent_tf_ptr = parent_tf.typed_data();
    const void* parent_idx_ptr = parent_idx.typed_data();
    const void* act_idx_ptr = act_idx.typed_data();
    const void* mimic_mul_ptr = mimic_mul.typed_data();
    const void* mimic_off_ptr = mimic_off.typed_data();
    const void* mimic_act_idx_ptr = mimic_act_idx.typed_data();
    const void* fk_level_starts_ptr = fk_level_starts.typed_data();
    const void* fk_level_joints_ptr = fk_level_joints.typed_data();
    const bool need_const_upload =
        !cache.valid ||
        cache.n_joints != n_joints ||
        cache.n_levels != n_levels ||
        cache.twists != twists_ptr ||
        cache.parent_tf != parent_tf_ptr ||
        cache.parent_idx != parent_idx_ptr ||
        cache.act_idx != act_idx_ptr ||
        cache.mimic_mul != mimic_mul_ptr ||
        cache.mimic_off != mimic_off_ptr ||
        cache.mimic_act_idx != mimic_act_idx_ptr ||
        cache.fk_level_starts != fk_level_starts_ptr ||
        cache.fk_level_joints != fk_level_joints_ptr;
    if (need_const_upload) {
        if (!copy_const(twists_ptr, c_twists, sizeof(float) * n_joints * 6) ||
            !copy_const(parent_tf_ptr, c_parent_tf, sizeof(float) * n_joints * 7) ||
            !copy_const(parent_idx_ptr, c_parent_idx, sizeof(int) * n_joints) ||
            !copy_const(act_idx_ptr, c_act_idx, sizeof(int) * n_joints) ||
            !copy_const(mimic_mul_ptr, c_mimic_mul, sizeof(float) * n_joints) ||
            !copy_const(mimic_off_ptr, c_mimic_off, sizeof(float) * n_joints) ||
            !copy_const(mimic_act_idx_ptr, c_mimic_act_idx, sizeof(int) * n_joints) ||
            !copy_const(fk_level_starts_ptr, c_fk_level_starts, sizeof(int) * (n_levels + 1)) ||
            !copy_const(fk_level_joints_ptr, c_fk_level_joints, sizeof(int) * n_joints)) {
            return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(cudaGetLastError()));
        }

        // Compute max level width from fk_level_starts (D2H copy, once per model).
        int h_level_starts[FK_MAX_JOINTS + 1];
        {
            const cudaError_t e = cudaMemcpyAsync(
                h_level_starts,
                fk_level_starts_ptr,
                sizeof(int) * (n_levels + 1),
                cudaMemcpyDeviceToHost,
                stream);
            if (e != cudaSuccess)
                return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
            cudaStreamSynchronize(stream);
        }
        int mlw = 0;
        for (int lvl = 0; lvl < n_levels; ++lvl) {
            const int w = h_level_starts[lvl + 1] - h_level_starts[lvl];
            if (w > mlw) mlw = w;
        }

        cache.twists = twists_ptr;
        cache.parent_tf = parent_tf_ptr;
        cache.parent_idx = parent_idx_ptr;
        cache.act_idx = act_idx_ptr;
        cache.mimic_mul = mimic_mul_ptr;
        cache.mimic_off = mimic_off_ptr;
        cache.mimic_act_idx = mimic_act_idx_ptr;
        cache.fk_level_starts = fk_level_starts_ptr;
        cache.fk_level_joints = fk_level_joints_ptr;
        cache.n_joints = n_joints;
        cache.n_levels = n_levels;
        cache.max_level_width = mlw;
        cache.valid = true;
    }

    (void)topo_inv;

    const int max_level_width = cache.max_level_width;

    // Select ITEMS_PER_WARP: pack as many batch items per warp as the widest
    // level allows.  LANES_PER_ITEM = 32 / ITEMS_PER_WARP must be >= max_level_width.
    if (max_level_width <= 2)
        return launch_fk_kernel_warp<256, 16>(stream, cfg.typed_data(), out->typed_data(), batch, n_joints, n_act, n_levels);
    else if (max_level_width <= 4)
        return launch_fk_kernel_warp<256,  8>(stream, cfg.typed_data(), out->typed_data(), batch, n_joints, n_act, n_levels);
    else if (max_level_width <= 8)
        return launch_fk_kernel_warp<256,  4>(stream, cfg.typed_data(), out->typed_data(), batch, n_joints, n_act, n_levels);
    else if (max_level_width <= 16)
        return launch_fk_kernel_warp<256,  2>(stream, cfg.typed_data(), out->typed_data(), batch, n_joints, n_act, n_levels);
    else
        return launch_fk_kernel_warp<256,  1>(stream, cfg.typed_data(), out->typed_data(), batch, n_joints, n_act, n_levels);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FkCudaFfi, FkCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // cfg
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_level_starts
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_level_joints
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out
