#pragma once

// Use relative path here for runtime compilation
#include "quadrants/inc/constants.h"
#include <cstdint>

#if defined(QD_RUNTIME_HOST)
namespace quadrants::lang {
#endif

struct LLVMRuntime;
// "RuntimeContext" holds necessary data for kernel body execution, such as a
// pointer to the LLVMRuntime struct, kernel arguments, and the thread id (if on
// CPU).
struct RuntimeContext {
  char *arg_buffer{nullptr};

  LLVMRuntime *runtime{nullptr};

  int32_t cpu_thread_id;

  // Set to 1 by quadrants_assert_format_ctx when a runtime assertion (e.g.
  // out-of-bounds check) fails on CPU.  The codegen emits an early return
  // after each assert call when this is set, and the task runner breaks out
  // of its loop.
  //
  // NOTE: paired with `cpu_thread_id` above on purpose: putting the two
  // 4-byte fields back-to-back keeps the struct at a natural 32-byte
  // 8-aligned layout. Splitting them with the trailing 8-byte pointer
  // would force clang to emit the bitcode struct as a *packed* LLVM type
  // with trailing tail padding (`align 1`), which on AMDGPU collapses the
  // kernarg-by-value copy into byte stores and regresses every launch.
  int32_t cpu_assert_failed{0};

  // We move the pointer of result buffer from LLVMRuntime to RuntimeContext
  // because each real function need a place to store its result, but
  // LLVMRuntime is shared among functions. So we moved the pointer to
  // RuntimeContext which each function have one.
  uint64_t *result_buffer;
};

#if defined(QD_RUNTIME_HOST)
}  // namespace quadrants::lang
#endif
