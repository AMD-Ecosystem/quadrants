#include "quadrants/runtime/amdgpu/amdgpu_graph_manager.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"

#include <cstring>
#include <vector>

namespace quadrants::lang {
namespace amdgpu {

namespace {

// Inline copy of KernelLauncher::on_amdgpu_device. Free function here to
// avoid a circular include between kernel_launcher and graph_manager.
// hipPointerGetAttributes returns HIP_SUCCESS + memoryType==Device for
// genuinely device-resident pointers.
bool ptr_on_amdgpu_device(void *ptr) {
  unsigned int attr_val[8];
  uint32_t ret_code =
      AMDGPUDriver::get_instance().mem_get_attributes.call(attr_val, ptr);
  return ret_code == HIP_SUCCESS && attr_val[0] == HIP_MEMORYTYPE_DEVICE;
}

}  // namespace

CachedAmdgpuGraph::~CachedAmdgpuGraph() {
  if (graph_exec) {
    AMDGPUDriver::get_instance().graph_exec_destroy(graph_exec);
  }
  if (persistent_device_arg_buffer) {
    AMDGPUDriver::get_instance().mem_free(persistent_device_arg_buffer);
  }
  if (persistent_device_result_buffer) {
    AMDGPUDriver::get_instance().mem_free(persistent_device_result_buffer);
  }
}

CachedAmdgpuGraph::CachedAmdgpuGraph(CachedAmdgpuGraph &&other) noexcept
    : graph_exec(other.graph_exec),
      persistent_device_arg_buffer(other.persistent_device_arg_buffer),
      persistent_device_result_buffer(other.persistent_device_result_buffer),
      persistent_ctx(other.persistent_ctx),
      arg_buffer_size(other.arg_buffer_size),
      result_buffer_size(other.result_buffer_size),
      num_nodes(other.num_nodes) {
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
}

CachedAmdgpuGraph &CachedAmdgpuGraph::operator=(
    CachedAmdgpuGraph &&other) noexcept {
  if (this != &other) {
    if (graph_exec)
      AMDGPUDriver::get_instance().graph_exec_destroy(graph_exec);
    if (persistent_device_arg_buffer)
      AMDGPUDriver::get_instance().mem_free(persistent_device_arg_buffer);
    if (persistent_device_result_buffer)
      AMDGPUDriver::get_instance().mem_free(persistent_device_result_buffer);

    graph_exec = other.graph_exec;
    persistent_device_arg_buffer = other.persistent_device_arg_buffer;
    persistent_device_result_buffer = other.persistent_device_result_buffer;
    persistent_ctx = other.persistent_ctx;
    arg_buffer_size = other.arg_buffer_size;
    result_buffer_size = other.result_buffer_size;
    num_nodes = other.num_nodes;

    other.graph_exec = nullptr;
    other.persistent_device_arg_buffer = nullptr;
    other.persistent_device_result_buffer = nullptr;
  }
  return *this;
}

// Resolves ndarray parameter handles in the launch context to raw device
// pointers, writing them into the arg buffer via set_ndarray_ptrs.
//
// Unlike the normal launch path, this does NOT handle host-resident arrays
// (no temporary device allocation or H2D transfer): the cached graph
// captures the arg buffer's device-pointer values at build time, so a
// host-resident array would mean the captured graph references a temporary
// host->device buffer that no longer exists by the next launch. Returns
// false to signal the caller should fall back to the normal launch path.
bool AmdgpuGraphManager::resolve_ctx_ndarray_ptrs(
    LaunchContextBuilder &ctx,
    const std::vector<std::pair<int, Callable::Parameter>> &parameters,
    LlvmRuntimeExecutor *executor) {
  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[arg_id];
      if (arr_sz == 0)
        continue;

      ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];

      // Graph path doesn't model autograd writeback, so refuse rather than
      // silently lose gradients. Caller falls back to normal launcher,
      // which has the gradient-fix code path.
      if (grad_ptr != nullptr) {
        QD_TRACE(
            "amdgpu_graph: declining (ndarray arg {} has non-null grad_ptr; "
            "graph path doesn't support autograd)",
            arg_id);
        return false;
      }

      void *resolved_data = nullptr;

      if (ctx.device_allocation_type[arg_id] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        if (!ptr_on_amdgpu_device(data_ptr)) {
          QD_TRACE(
              "amdgpu_graph: declining (ndarray arg {} is host-resident; "
              "graph path requires device-resident arrays)",
              arg_id);
          return false;
        }
        resolved_data = data_ptr;
      } else if (arr_sz > 0) {
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        resolved_data = executor->get_device_alloc_info_ptr(*ptr);
      }

      if (resolved_data) {
        ctx.set_ndarray_ptrs(arg_id, (uint64)resolved_data, (uint64) nullptr);
      }
    }
  }
  return true;
}

void *AmdgpuGraphManager::add_kernel_node(void *graph,
                                          void *prev_node,
                                          void *func,
                                          unsigned int grid_dim,
                                          unsigned int block_dim,
                                          unsigned int shared_mem,
                                          void **kernel_params) {
  AmdgpuKernelNodeParams params{};
  params.func = func;
  params.gridDimX = grid_dim;
  params.gridDimY = 1;
  params.gridDimZ = 1;
  params.blockDimX = block_dim;
  params.blockDimY = 1;
  params.blockDimZ = 1;
  params.sharedMemBytes = shared_mem;
  params.kernelParams = kernel_params;
  params.extra = nullptr;

  void *node = nullptr;
  AMDGPUDriver::get_instance().graph_add_kernel_node(
      &node, graph, prev_node ? &prev_node : nullptr, prev_node ? 1 : 0,
      &params);
  return node;
}

bool AmdgpuGraphManager::launch_cached_graph(CachedAmdgpuGraph &cached,
                                             LaunchContextBuilder &ctx) {
  // Update the persistent arg buffer in place so the cached graph (which
  // captured the buffer's address, not its contents) sees this launch's
  // arg values.
  if (ctx.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_host_to_device(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
        cached.arg_buffer_size);
  }
  // Default (NULL) HIP stream -- matches the rest of KernelLauncher's
  // stream affinity. When a per-handle stream is introduced later, plumb
  // it through here.
  AMDGPUDriver::get_instance().graph_launch(cached.graph_exec, nullptr);
  used_on_last_call_ = true;
  num_nodes_on_last_call_ = cached.num_nodes;
  return true;
}

bool AmdgpuGraphManager::try_launch(
    int launch_id,
    LaunchContextBuilder &ctx,
    JITModule *amdgpu_module,
    const std::vector<std::pair<int, Callable::Parameter>> &parameters,
    const std::vector<OffloadedTask> &offloaded_tasks,
    LlvmRuntimeExecutor *executor) {
  if (offloaded_tasks.empty()) {
    return false;
  }

  // v1 scope: graph_do_while requires HIP conditional nodes, which we
  // don't wire. Caller will fall through to the existing host-loop
  // implementation in launch_offloaded_tasks_with_do_while -- correct
  // but slower (each iteration round-trips to host to read the flag).
  if (ctx.graph_do_while_arg_id >= 0) {
    QD_TRACE(
        "amdgpu_graph: declining (graph_do_while not supported in v1; "
        "falling back to host-loop launcher)");
    return false;
  }

  // v1 scope: kernels that return result_buffer values would need a D2H
  // copy after the cached graph completes; the cached-graph path doesn't
  // model that. Genesis hot-loop kernels don't return values, so this
  // is fine for the actual use case.
  if (ctx.result_buffer_size > 0) {
    QD_TRACE(
        "amdgpu_graph: declining (kernel returns a value; "
        "graph path doesn't model result-buffer D2H)");
    return false;
  }

  // Walk the parameters; this either succeeds (every ndarray is on
  // device, no autograd) or returns false signalling a fall-back.
  if (!resolve_ctx_ndarray_ptrs(ctx, parameters, executor)) {
    return false;
  }

  auto it = cache_.find(launch_id);
  if (it != cache_.end()) {
    return launch_cached_graph(it->second, ctx);
  }

  AMDGPUContext::get_instance().make_current();

  CachedAmdgpuGraph cached;

  // --- Allocate persistent device buffers ---
  cached.result_buffer_size = std::max(ctx.result_buffer_size, sizeof(uint64));
  AMDGPUDriver::get_instance().malloc(
      (void **)&cached.persistent_device_result_buffer,
      cached.result_buffer_size);

  cached.arg_buffer_size = ctx.arg_buffer_size;
  if (cached.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().malloc(
        (void **)&cached.persistent_device_arg_buffer, cached.arg_buffer_size);
    AMDGPUDriver::get_instance().memcpy_host_to_device(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
        cached.arg_buffer_size);
  }

  // --- Build the persistent RuntimeContext that every captured kernel
  // node will receive as its first argument. The captured graph holds a
  // pointer to this ctx, so it must outlive the graph (it's a member of
  // CachedAmdgpuGraph; lives as long as the cache entry). ---
  cached.persistent_ctx.runtime = executor->get_llvm_runtime();
  cached.persistent_ctx.arg_buffer = cached.persistent_device_arg_buffer;
  cached.persistent_ctx.result_buffer =
      (uint64 *)cached.persistent_device_result_buffer;
  cached.persistent_ctx.cpu_thread_id = 0;

  // --- Build the HIP graph ---
  void *graph = nullptr;
  AMDGPUDriver::get_instance().graph_create(&graph, 0);

  void *prev_node = nullptr;
  for (const auto &task : offloaded_tasks) {
    void *ctx_ptr = &cached.persistent_ctx;
    prev_node = add_kernel_node(
        graph, prev_node, amdgpu_module->lookup_function(task.name),
        (unsigned int)task.grid_dim, (unsigned int)task.block_dim,
        (unsigned int)task.dynamic_shared_array_bytes, &ctx_ptr);
  }

  // --- Instantiate and launch ---
  AMDGPUDriver::get_instance().graph_instantiate(&cached.graph_exec, graph,
                                                 nullptr, nullptr, 0);

  AMDGPUDriver::get_instance().graph_launch(cached.graph_exec, nullptr);

  // The exec is independent of the template graph; we keep only the exec.
  AMDGPUDriver::get_instance().graph_destroy(graph);

  cached.num_nodes = offloaded_tasks.size();

  QD_TRACE("amdgpu_graph: created with {} kernel nodes for launch_id={}",
           cached.num_nodes, launch_id);

  num_nodes_on_last_call_ = cached.num_nodes;
  cache_.emplace(launch_id, std::move(cached));
  used_on_last_call_ = true;
  return true;
}

}  // namespace amdgpu
}  // namespace quadrants::lang
