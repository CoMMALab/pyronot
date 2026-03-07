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

// ---------------------------------------------------------------------------
// FK kernel
// ---------------------------------------------------------------------------

/**
 * One CUDA thread handles one batch element.
 * Delegates to fk_single() from _fk_cuda_helpers.cuh.
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
 * @param out            (batch, n_joints, 7)  float32  world transforms, orig-joint-indexed
 */
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
               float*       __restrict__ out,
               int batch, int n_joints, int n_act)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    fk_single(
        cfg    + (long long)b * n_act,
        twists, parent_tf, parent_idx, act_idx,
        mimic_mul, mimic_off, mimic_act_idx, topo_inv,
        out    + (long long)b * n_joints * 7,
        n_joints, n_act);
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
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out)
{
    const int batch    = static_cast<int>(cfg.dimensions()[0]);
    const int n_act    = static_cast<int>(cfg.dimensions()[1]);
    const int n_joints = static_cast<int>(twists.dimensions()[0]);

    constexpr int THREADS = 256;
    const int blocks = (batch + THREADS - 1) / THREADS;

    fk_kernel<<<blocks, THREADS, 0, stream>>>(
        cfg.typed_data(),
        twists.typed_data(),
        parent_tf.typed_data(),
        parent_idx.typed_data(),
        act_idx.typed_data(),
        mimic_mul.typed_data(),
        mimic_off.typed_data(),
        mimic_act_idx.typed_data(),
        topo_inv.typed_data(),
        out->typed_data(),
        batch,
        n_joints,
        n_act);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));

    return ffi::Error::Success();
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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()); // out
