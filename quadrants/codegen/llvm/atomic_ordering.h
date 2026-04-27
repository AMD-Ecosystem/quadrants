// Default LLVM atomic ordering used by Quadrants codegen for atomic RMW /
// compare-exchange.
//
// Background: prior to this knob, all atomic operations emitted by the
// Quadrants LLVM backend used `llvm::AtomicOrdering::SequentiallyConsistent`.
// On AMDGPU (gfx942) that lowering inserts two `s_waitcnt` fences and a
// buffer-invalidate around every atomic, which is overkill for the way the
// Quadrants DSL is used in practice (counters, accumulators, flag merges
// where synchronization happens at kernel-launch boundaries, not between
// individual atomics in a kernel).
//
// Default: `Monotonic` — single-location ordering only, no system fences.
//          This matches the semantics required by the Quadrants frontend
//          (`atomic_add`, `atomic_or`, etc. are reductions with no implied
//          cross-location ordering) and is the widely-used default in
//          GPU atomic codegen.
//
// Override: build with `-DQD_ATOMIC_ORDERING_SEQCST=1` to fall back to the
// previous SequentiallyConsistent behavior. This is provided as an A/B
// escape hatch in case any caller turns out to depend on the implicit
// system-scope fences.

#pragma once

#ifdef QD_WITH_LLVM

#include "llvm/IR/Instructions.h"

namespace quadrants::lang {

inline llvm::AtomicOrdering qd_default_atomic_ordering() {
#if defined(QD_ATOMIC_ORDERING_SEQCST) && QD_ATOMIC_ORDERING_SEQCST
  return llvm::AtomicOrdering::SequentiallyConsistent;
#else
  return llvm::AtomicOrdering::Monotonic;
#endif
}

}  // namespace quadrants::lang

#endif  // QD_WITH_LLVM
