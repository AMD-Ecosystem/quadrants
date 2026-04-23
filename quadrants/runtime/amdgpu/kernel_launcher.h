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

    // Cached per-launch device scratch buffers. Reused across launches to
    // avoid per-launch hipMallocAsync/hipFreeAsync (which call into
    // __amd_rocclr_copyBuffer-adjacent CLR machinery and also imply implicit
    // stream serialization). Lazily grown on demand; never shrunk;
    // explicitly freed in ~KernelLauncher() so that gs.destroy() (which
    // tears down ProgramImpl and thus the launcher unique_ptr) reclaims
    // them while the AMDGPU context is still alive. If the user never calls
    // destroy and the launcher reaches static destruction at process exit,
    // the AMDGPU context may already be torn down — in that case the OS
    // reclaims the device memory anyway and our destructor's mem_free_async
    // calls become best-effort.
    char *device_result_buffer{nullptr};
    size_t device_result_buffer_capacity{0};
    char *device_arg_buffer{nullptr};
    size_t device_arg_buffer_capacity{0};
  };

 public:
  using Base::Base;
  ~KernelLauncher() override;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;

 private:
  void launch_offloaded_tasks(
      LaunchContextBuilder &ctx,
      JITModule *amdgpu_module,
      const std::vector<OffloadedTask> &offloaded_tasks);
  void launch_offloaded_tasks_with_do_while(
      LaunchContextBuilder &ctx,
      JITModule *amdgpu_module,
      const std::vector<OffloadedTask> &offloaded_tasks);
  bool on_amdgpu_device(void *ptr);
  std::vector<Context> contexts_;
};

}  // namespace amdgpu
}  // namespace quadrants::lang
