#pragma once

#include <memory>
#include <mutex>

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
    // Per-handle persistent device arg buffer + cached host bytes. Replaces
    // the prior thread_local shared buffer so we can safely skip the H2D
    // when this handle's host arg buffer is byte-identical to its previous
    // launch (a common pattern: same kernel called every step with the same
    // ndarray pointers + scene-static scalars).
    char *dev_arg_buf{nullptr};
    std::size_t dev_arg_buf_cap{0};
    std::vector<char> last_host_arg_buf;
    // Serializes cache-check / H2D / cache-update so two host threads
    // launching the same handle concurrently can't race on dev_arg_buf
    // or last_host_arg_buf. unique_ptr keeps Context movable for the
    // contexts_ vector's resize / reallocation path.
    std::unique_ptr<std::mutex> arg_buf_mu = std::make_unique<std::mutex>();
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
