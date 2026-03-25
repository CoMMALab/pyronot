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

template <int THREADS>
__device__ __forceinline__
void fk_level_barrier() {
    if constexpr (THREADS == 32) {
        __syncwarp();
    } else {
        __syncthreads();
    }
}

__device__ __forceinline__
void fk_eval_joint(
    int j,
    const float* __restrict__ cfg_b,
    const float* __restrict__ s_twists,
    const float* __restrict__ s_parent_tf,
    const int*   __restrict__ s_parent_idx,
    const int*   __restrict__ s_act_idx,
    const float* __restrict__ s_mimic_mul,
    const float* __restrict__ s_mimic_off,
    const int*   __restrict__ s_mimic_act_idx,
    float*       __restrict__ out_b)
{
    const int m_idx = s_mimic_act_idx[j];
    const int a_idx = s_act_idx[j];
    const int src = (m_idx != -1) ? m_idx : a_idx;
    const float q_ref = (src == -1) ? 0.0f : cfg_b[src];
    const float q_j = q_ref * s_mimic_mul[j] + s_mimic_off[j];

    float tangent[6];
    #pragma unroll
    for (int k = 0; k < 6; ++k)
        tangent[k] = s_twists[j * 6 + k] * q_j;

    float delta_T[7];
    se3_exp(tangent, delta_T);

    float T_pc[7];
    se3_compose(s_parent_tf + j * 7, delta_T, T_pc);

    float* dst = out_b + j * 7;
    const int p = s_parent_idx[j];
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
 * One CUDA block handles one batch element.
 * Threads in the block process joints in parallel, grouped by depth level.
 *
 * @param cfg            (batch, n_act)        float32  actuated config
 * @param twists         (n_joints, 6)         float32  Lie-algebra twist / joint
 * @param parent_tf      (n_joints, 7)         float32  constant T_parent_joint [wxyz_xyz]
 * @param parent_idx     (n_joints,)           int32    original parent joint idx, -1 for roots
 * @param act_idx        (n_joints,)           int32    actuated joint source idx, -1 if fixed
 * @param mimic_mul      (n_joints,)           float32  mimic multiplier (1.0 for non-mimic)
 * @param mimic_off      (n_joints,)           float32  mimic offset (0.0 for non-mimic)
 * @param mimic_act_idx  (n_joints,)           int32    mimicked actuated idx, -1 if not mimic
 * @param topo_inv       (n_joints,)           int32    topo_sort_inv: sorted_i -> orig_j
 * @param fk_level_starts (n_levels + 1,)      int32    offsets into fk_level_joints
 * @param fk_level_joints (n_joints,)          int32    joint indices grouped by level
 * @param out            (batch, n_joints, 7)  float32  world transforms, orig-joint-indexed
 */
template <int THREADS>
__global__
void fk_kernel(const float* __restrict__ cfg,
               const float* __restrict__ twists,
               const float* __restrict__ parent_tf,
               const int*   __restrict__ parent_idx,
               const int*   __restrict__ act_idx,
               const float* __restrict__ mimic_mul,
               const float* __restrict__ mimic_off,
               const int*   __restrict__ mimic_act_idx,
               const int*   __restrict__ topo_inv,
               const int*   __restrict__ fk_level_starts,
               const int*   __restrict__ fk_level_joints,
               float*       __restrict__ out,
               int batch, int n_joints, int n_act, int n_levels)
{
    // ── Shared memory: robot parameters loaded once per block ───────────────
    // With n_joints ≤ FK_MAX_JOINTS=64 the footprint is ~3.7 KB/block, well
    // within the 48 KB limit.  All threads in the block collaborate on
    // the initial load so every FK call reads from L1-backed shared memory
    // instead of global DRAM.  The early-exit guard comes AFTER __syncthreads
    // so out-of-range threads still contribute to the cooperative load.
    __shared__ float s_twists       [FK_MAX_JOINTS * 6];
    __shared__ float s_parent_tf    [FK_MAX_JOINTS * 7];
    __shared__ int   s_parent_idx   [FK_MAX_JOINTS];
    __shared__ int   s_act_idx      [FK_MAX_JOINTS];
    __shared__ float s_mimic_mul    [FK_MAX_JOINTS];
    __shared__ float s_mimic_off    [FK_MAX_JOINTS];
    __shared__ int   s_mimic_act_idx[FK_MAX_JOINTS];
    __shared__ int   s_fk_level_starts[FK_MAX_JOINTS + 1];
    __shared__ int   s_fk_level_joints[FK_MAX_JOINTS];

    (void)topo_inv;

    for (int i = threadIdx.x; i < n_joints * 6; i += THREADS) s_twists[i]    = twists[i];
    for (int i = threadIdx.x; i < n_joints * 7; i += THREADS) s_parent_tf[i] = parent_tf[i];
    for (int i = threadIdx.x; i < n_joints;     i += THREADS) {
        s_parent_idx[i]    = parent_idx[i];
        s_act_idx[i]       = act_idx[i];
        s_mimic_mul[i]     = mimic_mul[i];
        s_mimic_off[i]     = mimic_off[i];
        s_mimic_act_idx[i] = mimic_act_idx[i];
        s_fk_level_joints[i] = fk_level_joints[i];
    }
    for (int i = threadIdx.x; i < n_levels + 1; i += THREADS)
        s_fk_level_starts[i] = fk_level_starts[i];
    __syncthreads();  // Ensure all robot data is visible before FK begins.

    const int start_b = blockIdx.x;
    const int stride_b = gridDim.x;

    for (int b = start_b; b < batch; b += stride_b) {
        const float* cfg_b = cfg + (long long)b * n_act;
        float* out_b = out + (long long)b * n_joints * 7;

        for (int lvl = 0; lvl < n_levels; ++lvl) {
            const int begin = s_fk_level_starts[lvl];
            const int end   = s_fk_level_starts[lvl + 1];
            const int width = end - begin;

            // Fast path: one thread maps to one joint in this level.
            if (width <= THREADS) {
                if (threadIdx.x < width) {
                    const int j = s_fk_level_joints[begin + threadIdx.x];
                    fk_eval_joint(
                        j, cfg_b, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                        s_mimic_mul, s_mimic_off, s_mimic_act_idx, out_b);
                }
            } else {
                // Fallback for unusually wide levels.
                for (int idx = begin + threadIdx.x; idx < end; idx += THREADS) {
                    const int j = s_fk_level_joints[idx];
                    fk_eval_joint(
                        j, cfg_b, s_twists, s_parent_tf, s_parent_idx, s_act_idx,
                        s_mimic_mul, s_mimic_off, s_mimic_act_idx, out_b);
                }
            }
            fk_level_barrier<THREADS>();
        }
    }
}

template <int THREADS>
static inline ffi::Error launch_fk_kernel(
    cudaStream_t stream,
    const float* cfg,
    const float* twists,
    const float* parent_tf,
    const int* parent_idx,
    const int* act_idx,
    const float* mimic_mul,
    const float* mimic_off,
    const int* mimic_act_idx,
    const int* topo_inv,
    const int* fk_level_starts,
    const int* fk_level_joints,
    float* out,
    int batch,
    int n_joints,
    int n_act,
    int n_levels)
{
    const int blocks = (batch < 65535) ? batch : 65535;
    fk_kernel<THREADS><<<blocks, THREADS, 0, stream>>>(
        cfg,
        twists,
        parent_tf,
        parent_idx,
        act_idx,
        mimic_mul,
        mimic_off,
        mimic_act_idx,
        topo_inv,
        fk_level_starts,
        fk_level_joints,
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

    // Select a compact launch width for FK_MAX_JOINTS <= 64.
    if (n_joints <= 32) {
        return launch_fk_kernel<32>(
            stream,
            cfg.typed_data(),
            twists.typed_data(),
            parent_tf.typed_data(),
            parent_idx.typed_data(),
            act_idx.typed_data(),
            mimic_mul.typed_data(),
            mimic_off.typed_data(),
            mimic_act_idx.typed_data(),
            topo_inv.typed_data(),
            fk_level_starts.typed_data(),
            fk_level_joints.typed_data(),
            out->typed_data(),
            batch,
            n_joints,
            n_act,
            n_levels);
    }
    return launch_fk_kernel<64>(
        stream,
        cfg.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        fk_level_starts.typed_data(),
        fk_level_joints.typed_data(),
        out->typed_data(),
        batch,
        n_joints,
        n_act,
        n_levels);
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
