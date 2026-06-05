#pragma once

#include <deque>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/amdgpu/graph_manager.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace amdgpu {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct Context {
    JITModule *jit_module{nullptr};
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
    std::vector<OffloadedTask> offloaded_tasks;
    // Per-kernel-handle persistent scratch. Allocated lazily on first use, grown amortised-doubling on demand,
    // freed only on launcher destruction. Stored per-handle (not per-launcher) because `publish_adstack_metadata`'s
    // host-eval branch evaluates `FieldLoad` SizeExpr leaves through `SNodeRwAccessorsBank`, which recursively
    // launches `snode_reader_*` kernels via the same launcher singleton; sharing one device-side `arg_buffer` /
    // `RuntimeContext` across kernels would let the recursive launch overwrite the parent's args before the parent
    // kernel reads them. Per-handle buffers give each kernel a private device address that recursive launches
    // cannot touch.
    void *arg_buffer_dev_ptr{nullptr};
    std::size_t arg_buffer_capacity{0};
    void *runtime_context_dev_ptr{nullptr};
    // Per-task device-function-pointer cache. `JITModuleAMDGPU::lookup_function` is itself memoized but each call
    // still takes a mutex + hash-table lookup. With Genesis dispatching ~100 kernels per step, skipping that work
    // is a measurable host-side win. The vector is sized lazily on first launch and reused across launches.
    std::vector<void *> resolved_funcs;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(const LLVM::CompiledKernelData &compiled) override;
  std::size_t get_graph_cache_size() const override {
    return graph_manager_.cache_size();
  }
  bool get_graph_cache_used_on_last_call() const override {
    return graph_manager_.used_on_last_call();
  }
  std::size_t get_graph_num_nodes_on_last_call() const override {
    return graph_manager_.num_nodes_on_last_call();
  }
  std::size_t get_graph_total_builds() const override {
    return graph_manager_.total_builds();
  }

 private:
  // `kernarg_payload` / `kernarg_size`: bytes to feed the kernel as its single argument.
  //   AMDGPU's `kernel_argument_struct_in_kernarg() == true` codegen path uses this to receive the `RuntimeContext`
  //   struct **by value** through the AMDGPU kernarg cache, eliminating both the per-launch `RuntimeContext` H2D
  //   memcpy and the per-instruction kernel loads through a `RuntimeContext*`. The host pointer
  //   (`&ctx.get_context()`) is read by `AMDGPUContext::launch` while staging the AQL kernarg packet, so no device
  //   allocation is required for forward-only kernels.
  // `device_runtime_context_ptr_for_adstack`: device-resident `RuntimeContext` shadow used by
  //   `publish_adstack_metadata` / `publish_per_task_bound_count_device`. Only autodiff kernels that have at least
  //   one task with `task.ad_stack.allocas != {}` need this; forward-only kernels (Genesis hot path) pass nullptr
  //   and skip the allocate + H2D entirely.
  void launch_offloaded_tasks(LaunchContextBuilder &ctx,
                              JITModule *amdgpu_module,
                              const std::vector<OffloadedTask> &offloaded_tasks,
                              std::vector<void *> &resolved_funcs,
                              void *kernarg_payload,
                              int kernarg_size,
                              void *device_runtime_context_ptr_for_adstack);
  void launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                            JITModule *amdgpu_module,
                                            const std::vector<OffloadedTask> &offloaded_tasks,
                                            std::vector<void *> &resolved_funcs,
                                            void *kernarg_payload,
                                            int kernarg_size,
                                            void *device_runtime_context_ptr_for_adstack);
  bool on_amdgpu_device(void *ptr);
  // `result_buffer` is the only scratch that stays launcher-global: kernels write to it then the host reads it
  // back via `hipMemcpyDtoH` before the next kernel runs, so recursive snode-reader launches that reuse it cannot
  // smuggle stale bytes into the parent's read (the parent's host-side `fetch_result_uint64` happens after every
  // child completes, before the parent kernel that would be the next reader). Grown amortised-doubling.
  void *persistent_result_buffer_dev_ptr_{nullptr};
  std::size_t persistent_result_buffer_capacity_{0};
  // std::deque (not std::vector): `publish_adstack_metadata`'s host-eval branch recursively registers snode-reader
  // kernels via this same launcher, calling `contexts_.resize()` while a parent `launch_llvm_kernel` frame still
  // holds a reference into the container.  std::deque never invalidates references on push_back / resize, so the
  // parent's `launcher_ctx` reference survives the child's registration.
  std::deque<Context> contexts_;
  GraphManager graph_manager_;

 public:
  ~KernelLauncher() override;
};

}  // namespace amdgpu
}  // namespace quadrants::lang
