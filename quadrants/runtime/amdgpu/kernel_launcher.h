#pragma once

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace amdgpu {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct Context {
    JITModule *jit_module{nullptr};
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
    std::vector<OffloadedTask> offloaded_tasks;
    std::vector<void *> resolved_funcs;
    // Per-handle persistent device-side arg_buffer scratch + byte-hash
    // cache. The pre-existing implementation used a single thread_local
    // buffer shared across all kernel handles; with the buffer shared,
    // hashing host bytes to skip duplicate H2D never wins because
    // consecutive Genesis launches hit different kernels with different
    // arg_buffers, so the cached hash never matches. Storing the buffer
    // per handle lets each kernel keep its own device address; consecutive
    // launches of the SAME kernel can then byte-compare and skip the H2D
    // when shape + ndarray-pointer slots are unchanged. Reset on
    // (re)allocation.
    void *arg_buffer_dev_ptr{nullptr};
    std::size_t arg_buffer_capacity{0};
    uint64_t arg_buffer_cached_hash{0};
    std::size_t arg_buffer_cached_size{0};
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;

 private:
  void launch_offloaded_tasks(LaunchContextBuilder &ctx, Context &launcher_ctx);
  void launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                            Context &launcher_ctx);
  bool on_amdgpu_device(void *ptr);
  std::vector<Context> contexts_;
};

}  // namespace amdgpu
}  // namespace quadrants::lang
