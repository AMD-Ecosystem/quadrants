#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

namespace quadrants::lang {
namespace amdgpu {

// Mirrors CUDA_KERNEL_NODE_PARAMS / hipKernelNodeParams. We define our own
// copy because Quadrants loads HIP dynamically rather than linking against
// it, so we don't have access to <hip/hip_runtime_api.h> here. Field order
// and types verified against ROCm 5.x+ public headers.
//
// Only the basic kernel-node variant is used; conditional / event / memset
// node variants are NOT supported in this v1 of AmdgpuGraphManager (HIP's
// conditional node API is much newer than the basic graph APIs and the
// generic node-params struct layout is not yet ABI-stable across ROCm
// point releases).
struct AmdgpuKernelNodeParams {
  void *func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  void **kernelParams;
  void **extra;
};

// One cached graph per registered kernel handle (launch_id). Lifecycle:
//
//   first launch_llvm_kernel for this handle:
//     - allocate persistent device arg/result buffers
//     - build an hipGraph_t with one kernel node per offloaded task
//     - hipGraphInstantiate -> hipGraphExec_t (saved here)
//     - hipGraphDestroy on the template graph (we keep only the exec)
//   subsequent launches:
//     - memcpy_host_to_device the new arg buffer into the persistent slot
//     - hipGraphLaunch the cached exec
//
// Move-only (RAII over hipGraphExec_t + hipMalloc'd buffers); the cache_
// map relies on move semantics on rehash.
struct CachedAmdgpuGraph {
  // hipGraphExec_t handle (typed as void* since the driver is loaded
  // dynamically). This is the instantiated, launchable form of the graph.
  void *graph_exec{nullptr};
  char *persistent_device_arg_buffer{nullptr};
  char *persistent_device_result_buffer{nullptr};
  RuntimeContext persistent_ctx{};
  std::size_t arg_buffer_size{0};
  std::size_t result_buffer_size{0};
  std::size_t num_nodes{0};

  CachedAmdgpuGraph() = default;
  ~CachedAmdgpuGraph();
  CachedAmdgpuGraph(const CachedAmdgpuGraph &) = delete;
  CachedAmdgpuGraph &operator=(const CachedAmdgpuGraph &) = delete;
  CachedAmdgpuGraph(CachedAmdgpuGraph &&other) noexcept;
  CachedAmdgpuGraph &operator=(CachedAmdgpuGraph &&other) noexcept;
};

// Direct port of CudaGraphManager. v1 scope:
//   - YES:  basic kernel-graph capture + cached replay across launches.
//           The graph is built once per launch_id, instantiated, and reused
//           for every subsequent launch -- only the arg buffer is updated
//           on each call.
//   - NO:   graph_do_while via conditional nodes. AMDGPU's conditional
//           graph node ABI is not yet stable across ROCm point releases.
//           When ctx.graph_do_while_arg_id >= 0, try_launch declines and
//           the caller falls through to the existing host-loop path
//           (KernelLauncher::launch_offloaded_tasks_with_do_while), which
//           is correct (just slower, since each iteration round-trips to
//           the host to read the condition flag).
//   - NO:   support for kernels that return result_buffer values. Returning
//           values needs a D2H copy after the graph completes, which the
//           cached-graph path doesn't model. Genesis's hot-loop kernels
//           don't return values, so this is fine for the actual use case.
//   - NO:   support for kernels with host-resident ndarrays. The graph
//           captures the device pointer at build time; if the array later
//           moves, the cached graph would dereference a stale address.
//           We surface a clear error rather than silently produce wrong
//           results.
class AmdgpuGraphManager {
 public:
  // Attempts to launch the kernel via a cached or newly built HIP graph.
  // Returns true on success; false if the graph path can't be used (e.g.
  // graph_do_while requested, host-resident ndarrays, kernel returns a
  // value) and the caller should fall back to the normal launch path.
  // Internally tracks whether the graph was used, queryable via
  // used_on_last_call().
  bool try_launch(
      int launch_id,
      LaunchContextBuilder &ctx,
      JITModule *amdgpu_module,
      const std::vector<std::pair<int, Callable::Parameter>> &parameters,
      const std::vector<OffloadedTask> &offloaded_tasks,
      LlvmRuntimeExecutor *executor);

  // For tests / diagnostics.
  void mark_not_used() {
    used_on_last_call_ = false;
    num_nodes_on_last_call_ = 0;
  }
  std::size_t cache_size() const {
    return cache_.size();
  }
  bool used_on_last_call() const {
    return used_on_last_call_;
  }
  std::size_t num_nodes_on_last_call() const {
    return num_nodes_on_last_call_;
  }

 private:
  bool launch_cached_graph(CachedAmdgpuGraph &cached,
                           LaunchContextBuilder &ctx);
  bool resolve_ctx_ndarray_ptrs(
      LaunchContextBuilder &ctx,
      const std::vector<std::pair<int, Callable::Parameter>> &parameters,
      LlvmRuntimeExecutor *executor);
  void *add_kernel_node(void *graph,
                        void *prev_node,
                        void *func,
                        unsigned int grid_dim,
                        unsigned int block_dim,
                        unsigned int shared_mem,
                        void **kernel_params);

  // Keyed by launch_id, which uniquely identifies a compiled kernel
  // variant (each template specialization gets its own launch_id).
  std::unordered_map<int, CachedAmdgpuGraph> cache_;
  bool used_on_last_call_{false};
  std::size_t num_nodes_on_last_call_{0};
};

}  // namespace amdgpu
}  // namespace quadrants::lang
