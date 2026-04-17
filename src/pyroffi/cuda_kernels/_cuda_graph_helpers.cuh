/**
 * _cuda_graph_helpers.cuh
 *
 * Lightweight CUDA graph cache for single-kernel IK FFI handlers.
 *
 * Usage in each FFI handler
 * ─────────────────────────
 *   static IkGraphCache s_cache;
 *
 *   if (!s_cache.shape_matches(...)) {
 *       s_cache.invalidate();
 *       s_cache.ensure_capture_stream();
 *       cudaStreamBeginCapture(s_cache.capture_stream, ...);
 *       my_kernel<<<grid, threads, 0, s_cache.capture_stream>>>(...);
 *       cudaStreamEndCapture(s_cache.capture_stream, &s_cache.graph);
 *       s_cache.finalize_capture(np, ns, na, nj, ne);
 *   } else {
 *       cudaKernelNodeParams kp = {};
 *       kp.func ... kp.kernelParams = kargs;
 *       cudaGraphExecKernelNodeSetParams(s_cache.exec, s_cache.kernel_node, &kp);
 *   }
 *   cudaGraphLaunch(s_cache.exec, xla_stream);
 *
 * Benefits
 * ────────
 *   - Amortises CPU-side kernel-launch overhead (driver API call, argument
 *     marshalling) to a single cudaGraphLaunch per IK call.
 *   - Allows the driver to apply graph-level optimisations (dependency
 *     analysis, memory layout) once at instantiation, not on every call.
 *   - Compatible with XLA's stream scheduling: the graph is launched on the
 *     XLA-provided stream so ordering with other XLA operations is preserved.
 *
 * Constraints
 * ───────────
 *   - Not thread-safe; assumes serialised calls per CUDA device (satisfied
 *     by JAX's single-stream-per-device model).
 *   - Requires CUDA ≥ 10.2 (cudaGraphExecKernelNodeSetParams).
 */

#pragma once
#include <cuda_runtime.h>

struct IkGraphCache {
    cudaGraphExec_t  exec           = nullptr;
    cudaGraphNode_t  kernel_node    = nullptr;
    cudaGraph_t      graph          = nullptr;  // kept alive for node queries
    cudaStream_t     capture_stream = nullptr;

    // Kernel launch params cached at capture time; reused when building the
    // cudaKernelNodeParams for cudaGraphExecKernelNodeSetParams.
    void*        func_ptr   = nullptr;
    dim3         grid_dim   = {1, 1, 1};
    dim3         block_dim  = {1, 1, 1};
    unsigned int shared_mem = 0;

    // Shape fingerprint — invalidated whenever any dimension changes.
    int n_problems = -1, n_seeds = -1, n_act = -1, n_joints = -1, n_ee = -1;

    bool shape_matches(int np, int ns, int na, int nj, int ne) const noexcept {
        return np == n_problems && ns == n_seeds &&
               na == n_act     && nj == n_joints && ne == n_ee;
    }

    /** Create the dedicated capture stream (idempotent). */
    cudaError_t ensure_capture_stream() noexcept {
        if (capture_stream) return cudaSuccess;
        return cudaStreamCreateWithFlags(&capture_stream,
                                         cudaStreamNonBlocking);
    }

    /** Destroy the previous graph/exec and reset all state. */
    void invalidate() noexcept {
        if (exec)  { cudaGraphExecDestroy(exec);  exec  = nullptr; }
        if (graph) { cudaGraphDestroy(graph);      graph = nullptr; }
        kernel_node = nullptr;
        func_ptr    = nullptr;
        n_problems = n_seeds = n_act = n_joints = n_ee = -1;
    }

    /**
     * Called immediately after cudaStreamEndCapture writes into this->graph.
     *
     * Extracts the single kernel node, caches its launch params (func ptr,
     * grid/block dims, shared-memory size), and instantiates the executable
     * graph.  Returns cudaSuccess on success.
     */
    cudaError_t finalize_capture(int np, int ns, int na, int nj, int ne) noexcept {
        // Single-kernel graph → exactly one node.
        size_t n_nodes = 1;
        cudaError_t e = cudaGraphGetNodes(graph, &kernel_node, &n_nodes);
        if (e != cudaSuccess) return e;
        if (n_nodes == 0)     return cudaErrorUnknown;

        // Cache kernel launch params for future parameter updates.
        cudaKernelNodeParams kp = {};
        e = cudaGraphKernelNodeGetParams(kernel_node, &kp);
        if (e != cudaSuccess) return e;
        func_ptr   = kp.func;
        grid_dim   = kp.gridDim;
        block_dim  = kp.blockDim;
        shared_mem = kp.sharedMemBytes;

        // Instantiate the executable graph.
        e = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        if (e != cudaSuccess) return e;

        n_problems = np; n_seeds = ns; n_act = na; n_joints = nj; n_ee = ne;
        return cudaSuccess;
    }
};
