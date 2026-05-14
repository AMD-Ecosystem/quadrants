// SPDX-License-Identifier: MIT
//
// fuse_offloaded_tasks.cpp - JIT-level fusion of adjacent OffloadedStmt
// range_for tasks.
//
// Motivation. After irpass::offload() splits a kernel into N OffloadedStmt
// blocks, each top-level parallel range_for becomes a separate GPU kernel
// dispatch at runtime. On AMDGPU (gfx942) the command queue is saturated
// during the Genesis hot path (99.99% busy in rocprofv3 kernel-trace).
// Every extra kernel dispatch consumes queue cycles even if the kernel
// itself is cheap. Reducing dispatch count = reducing queue work =
// throughput gain.
//
// What this pass does. Walks the root block of an already-offloaded
// kernel and fuses pairs of adjacent OffloadedStmt's that have:
//   1. task_type == range_for in both
//   2. identical launch bounds (begin, end, block_dim) - either static
//      (compile-time-known) or dynamic-via-gtmp (both offloads read
//      their bound from the same global temporary slot)
//   3. *disjoint* global access resources (ndarrays / SNodes / gtmp)
//
// Condition (3) is a sufficient (not necessary) safety check. Pre-fusion,
// the kernel boundary acts as a global barrier - thread T1 in task A
// finishes before any thread starts task B. Post-fusion, every thread
// runs A's body then B's body, but threads are not synchronised
// cross-grid. If A writes a location that B reads (or vice versa, or
// both write the same location), threads from different workgroups can
// race.
//
// Disjoint-resources is conservative but tractable and catches the
// common Genesis case: multiple back-to-back zero-init loops that each
// touch a different ndarray (e.g. clear links_state.vel, clear
// dofs_state.acc, clear ...). Those produce one kernel dispatch each
// today; this pass fuses them.
//
// Same-thread RAW relaxation (QD_AGGRESSIVE_KERNEL_FUSION_RAW=1). When two
// adjacent offloads share a resource we additionally check whether *every*
// access to that resource (in either body) uses the same per-thread
// address fingerprint AND that address is a *provably injective*
// function of the loop index. Injectivity here means the mapping
// `i -> address(i)` is one-to-one, so thread T touches a byte that
// no other thread touches in either body - the cross-kernel-boundary
// RAW becomes an in-thread RAW which is naturally ordered by program
// order.
//
// Pure "depends on the loop index" is *not* sufficient. Non-injective
// patterns like `arr[i // 2]`, `arr[i % 2]`, `arr[min(i, K)]`, or
// `arr[i & 1]` all syntactically contain a loop index but collapse
// multiple threads onto the same byte. Pre-fusion the kernel boundary
// serialises the resulting cross-thread race; post-fusion it does not,
// so the observable result can change even though the user code was
// already racy.
//
// We therefore classify each sub-expression in the address as one of:
//   - constant (Const / Arg / kernel-launch-invariant gtmp / ATB)
//   - provably injective in the loop index (LoopIndexStmt, addition or
//     subtraction with a compile-time constant, multiplication by a
//     non-zero compile-time constant, shift-left by a compile-time
//     constant, XOR with a compile-time constant, negation, bit_not,
//     bitcast - all bijections on the integer lane)
//   - non-injective / unclassified (everything else, including div,
//     mod, max, min, bit_and, bit_or, shift-right, casts, math ops)
// An indexed access (ExternalPtr / GlobalPtr / MatrixPtr) is injective
// iff at least one of its index expressions is independently injective.
// LoopIndexStmts are canonicalised across offloads (so A's and B's loop
// indices fingerprint identically). Unhandled Stmt kinds fall back to
// pointer identity, which mismatches across A and B and forces a safe
// rejection.
//
// Sequencing. The pass *must* run after FixCrossOffloadReferences in
// irpass::offload() so that OffloadedStmt::begin_offset / end_offset
// are populated with their final gtmp byte offsets. Run earlier and
// the dynamic-bound matcher will see end_offset == 0 (the default
// value) on every dynamic offload and incorrectly fuse unrelated
// tasks, producing GPU memory faults.
//
// Disabled by default. Set QD_AGGRESSIVE_KERNEL_FUSION=1 (or on/true/
// yes) to enable the pass; this is an opt-in feature - workloads must
// validate fusion against their own correctness and perf criteria
// before enabling. With the master flag on, set
// QD_AGGRESSIVE_KERNEL_FUSION_RAW=0 to keep fusion on but disable the
// same-thread RAW relaxation, falling back to the disjoint-resources
// policy only. Set QD_AGGRESSIVE_KERNEL_FUSION_DIAG=1 to print per-
// kernel fusion decisions and reject reasons to stderr.

#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/program/kernel.h"
#include "quadrants/system/profiler.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace quadrants::lang {

namespace {

// Coarse-grained handle for a global resource that an IR statement
// touches. Two statements that produce the same Resource may alias;
// statements producing different Resources are guaranteed to be
// disjoint.
struct Resource {
  enum class Kind : int {
    Unknown = 0,
    Ndarray = 1,
    SNode = 2,
    GlobalTmp = 3,
  };
  Kind kind;
  int ndarray_arg_id;
  int ndarray_is_grad;
  SNode *snode;
  int64_t gtmp_offset;

  bool operator==(const Resource &o) const {
    if (kind != o.kind)
      return false;
    switch (kind) {
      case Kind::Unknown:
        return false;  // Unknowns alias with everything; never equal.
      case Kind::Ndarray:
        return ndarray_arg_id == o.ndarray_arg_id &&
               ndarray_is_grad == o.ndarray_is_grad;
      case Kind::SNode:
        return snode == o.snode;
      case Kind::GlobalTmp:
        return gtmp_offset == o.gtmp_offset;
    }
    return false;
  }
};

struct ResourceHash {
  size_t operator()(const Resource &r) const {
    size_t h = static_cast<size_t>(r.kind);
    switch (r.kind) {
      case Resource::Kind::Ndarray:
        h ^= std::hash<int>{}(r.ndarray_arg_id) + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(r.ndarray_is_grad) + (h << 6) + (h >> 2);
        break;
      case Resource::Kind::SNode:
        h ^= std::hash<const void *>{}(r.snode) + (h << 6) + (h >> 2);
        break;
      case Resource::Kind::GlobalTmp:
        h ^= std::hash<int64_t>{}(r.gtmp_offset) + (h << 6) + (h >> 2);
        break;
      default:
        break;
    }
    return h;
  }
};

// True iff `s` is a compile-time ConstStmt.
inline bool is_compile_time_const(Stmt *s) {
  return s != nullptr && s->is<ConstStmt>();
}

// True iff `s` is a compile-time ConstStmt whose numeric value is non-zero.
// Used by the injectivity classifier: multiplying or shift-left'ing an
// injective expression by a *non-zero* compile-time constant preserves
// injectivity; multiplying by zero collapses every thread onto address 0.
// equal_value(static_cast<int64_t>(0)) constructs a TypedConstant of the
// ConstStmt's own dtype with value 0 and compares - handles all integer
// and float primitive types correctly.
inline bool is_nonzero_compile_time_const(Stmt *s) {
  auto *c = s ? s->cast<ConstStmt>() : nullptr;
  if (!c)
    return false;
  return !c->val.equal_value(static_cast<int64_t>(0));
}

// Per-thread address fingerprint for an access.
//   is_injective_in_loop_idx : the fingerprinted expression is a provably
//                  injective function of the loop index - the mapping
//                  i -> address(i) is one-to-one, so different threads
//                  necessarily touch different bytes. This is strictly
//                  stronger than "address syntactically depends on the
//                  loop index": `arr[i // 2]` depends on `i` but is not
//                  injective in it. The same-thread RAW relaxation
//                  requires injectivity, not just dependence.
//   is_indirect  : the address dereferences another global pointer
//                  (loads from an ndarray / snode) as part of its
//                  index computation, e.g. `arr[other_arr[i], j]`.
//                  Such addresses are NOT provably injective in the
//                  loop index across threads (the indirection array
//                  could map different threads to the same byte), so
//                  even if A's and B's fingerprints match the
//                  same-thread RAW check has to reject - we can't
//                  rule out a cross-thread RAW.
struct AccessFp {
  std::string fp;
  bool is_injective_in_loop_idx;
  bool is_indirect;
  bool operator==(const AccessFp &o) const {
    return is_injective_in_loop_idx == o.is_injective_in_loop_idx &&
           is_indirect == o.is_indirect && fp == o.fp;
  }
};

struct AccessFpHash {
  size_t operator()(const AccessFp &a) const {
    return std::hash<std::string>{}(a.fp) ^
           (a.is_injective_in_loop_idx ? 0x9e3779b97f4a7c15ULL : 0) ^
           (a.is_indirect ? 0xc2b2ae3d27d4eb4fULL : 0);
  }
};

// Per-LocalLoadStmt resolution: the value most recently stored to its
// alloca at the point of the load, in body execution order. Computed
// per offload body so the fingerprinter can see through
// `LocalLoad(alloca)` to the expression that originally produced the
// value (typically a LoopIndexStmt or an arithmetic expression on
// one).
//
// We can't use a simpler "unique writer per alloca" map because
// Genesis emits an init store followed by a real store for almost
// every scalar local, e.g.
//   alloca i;
//   i = 0;                  // init
//   i = LoopIdx / N;        // real
//   ...arr[i] = ...
// "unique writer" treats `i` as multi-writer and gives up. The right
// answer for each LocalLoad is the most recent LocalStoreStmt to that
// alloca in program order; that's what this map captures.
using LoadMap = std::unordered_map<LocalLoadStmt *, Stmt *>;

// Recursively fingerprint an address-producing Stmt for the same-
// thread RAW check. The goal: two accesses fingerprint identically
// iff they touch the same byte for the same thread index in the
// fused kernel.
//
// Canonicalisation rules:
//   LoopIndexStmt(loop=*, index=k) -> "L{k}"  (offload identity dropped:
//     after fusion the two loop indices are unified via
//     replace_all_usages_with).
//   ConstStmt                      -> "C{stringified value}"
//   ArgLoadStmt                    -> "Arg{arg_id[0]}"  (kernel-launch
//     invariant, identical across A and B).
//   ExternalTensorBasePtrStmt      -> "ATB{arg_id[0]}{grad?}"
//   GlobalTemporaryStmt(offset=O)  -> "GT{O}"  (same slot in both
//     offloads).
//   LocalLoadStmt(src=AllocaStmt)  -> fingerprint of the alloca's
//     unique writer, if any (otherwise pointer fallback). This is the
//     escape hatch for the very common Genesis pattern:
//       loop_idx -> alloca -> LocalLoad -> used as index
//     Without it most physics-body accesses fingerprint as unrelated
//     even though they all derive from the loop index.
//   GlobalLoadStmt(src=...)        -> "R({fp(src)})"  (value-based;
//     caller is responsible for proving the source slot isn't written
//     between the two reads, which we enforce by also tracking the
//     load's resource as part of CollectGlobalAccesses - if either
//     body writes it, the resource conflict check will fail anyway).
//   ExternalPtrStmt                -> "EP({fp(base)},[i0,i1,...]{G?})"
//   GlobalPtrStmt                  -> "GP({snode_ptr},[i0,i1,...])"
//   MatrixPtrStmt                  -> "MP({fp(origin)},{fp(offset)})"
//   BinaryOpStmt                   -> "B{op}({fp(lhs)},{fp(rhs)})"
//   UnaryOpStmt                    -> "U{op}({fp(operand)})"
//   anything else                  -> "@{type}_{pointer}"  (conservative
//     fallback; two semantically-equal Stmts with different pointers
//     fingerprint as unequal, which forces a mismatch and rejects
//     fusion - safe).
//
// Injectivity tracking (out_is_injective_in_loop_idx):
//   The caller's flag is set to true iff the expression *as a whole* is
//   a provably-injective function of the loop index. Recursion uses
//   *local* bools and combines them with per-operator rules:
//     - LoopIndexStmt                       : injective.
//     - add/sub  i, c   (c compile-time const) : injective if i is.
//     - mul      i, c   (c non-zero const)    : injective if i is.
//     - bit_shl  i, c   (c const)            : injective if i is.
//     - bit_xor  i, c   (c const)            : injective if i is.
//     - neg / bit_not / cast_bits            : injective if operand is.
//     - EP / GP / MP                         : injective if at least one
//                                              index is independently
//                                              injective.
//     - LocalLoad (resolved)                 : passes through to the
//                                              stored value.
//     - everything else                      : NOT classified injective
//                                              (div, mod, max, min,
//                                              bit_and, bit_or, shr,
//                                              cmp, cast_value, math).
//   We never OR-cumulate the flag across unrelated subtrees, because
//   `(i + 1) / 2` is *not* injective even though `(i + 1)` is.
//
// depth is bounded to prevent runaway recursion on pathological IR.
// `inside_index` is true when fingerprinting an index sub-expression
// of an outer ExternalPtrStmt / GlobalPtrStmt / MatrixPtrStmt. Loads
// from another ndarray / snode inside an index are flagged via
// `out_is_indirect` so the caller can reject same-thread RAW for
// indirected accesses.
std::string fingerprint_value(Stmt *s,
                              int depth,
                              bool &out_is_injective_in_loop_idx,
                              bool &out_is_indirect,
                              bool inside_index,
                              const LoadMap *load_map) {
  if (depth > 16) {
    return "?";
  }
  if (!s) {
    return "N";
  }
  // LoopIndexStmt is the base case: thread T contributes exactly one
  // value of L, and this is the only "interesting" injective leaf.
  if (auto *li = s->cast<LoopIndexStmt>()) {
    out_is_injective_in_loop_idx = true;
    return "L" + std::to_string(li->index);
  }
  // Constants and launch-invariant values. These do not depend on the
  // loop index, so they never set out_is_injective_in_loop_idx. They are
  // still useful "neutral" operands for injectivity-preserving ops like
  // add-by-const and mul-by-const above.
  if (auto *c = s->cast<ConstStmt>()) {
    return "C" + c->val.stringify();
  }
  if (auto *al = s->cast<ArgLoadStmt>()) {
    int aid = al->arg_id.empty() ? -1 : al->arg_id[0];
    return "Arg" + std::to_string(aid);
  }
  if (auto *etbp = s->cast<ExternalTensorBasePtrStmt>()) {
    int aid = etbp->arg_id.empty() ? -1 : etbp->arg_id[0];
    return "ATB" + std::to_string(aid) + (etbp->is_grad ? "G" : "");
  }
  if (auto *gt = s->cast<GlobalTemporaryStmt>()) {
    return "GT" + std::to_string(gt->offset);
  }
  // LocalLoad of a resolved alloca is *transparent* - its injectivity
  // class is exactly the resolved value's class. Pass the caller's
  // flag through by reference so a loop-index-derived alloca slot can
  // be classified as injective.
  if (auto *ll = s->cast<LocalLoadStmt>()) {
    if (load_map) {
      auto it = load_map->find(ll);
      if (it != load_map->end()) {
        return fingerprint_value(it->second, depth + 1,
                                 out_is_injective_in_loop_idx, out_is_indirect,
                                 inside_index, load_map);
      }
    }
    // Could not resolve. Emit a stable token keyed on the alloca/src
    // pointer; two unresolved LocalLoads from the same alloca within
    // the same body fingerprint identically (still pointer-based, so
    // doesn't match across A and B). Use a throwaway local flag for
    // any recursion into ll->src so an unresolved load can't leak a
    // bogus injectivity claim to the caller.
    std::ostringstream oss;
    if (auto *alloca = ll->src->cast<AllocaStmt>()) {
      oss << "LL_alloca_" << reinterpret_cast<uintptr_t>(alloca);
    } else {
      bool ll_inj_local = false;
      oss << "LL("
          << fingerprint_value(ll->src, depth + 1, ll_inj_local,
                               out_is_indirect, inside_index, load_map)
          << ")";
    }
    return oss.str();
  }
  // GlobalLoadStmt: the loaded value is a runtime quantity. Even if it
  // is derived from an injective address inside its src, the loaded
  // *value* is not a statically known function of the loop index and
  // is not injective in it in general. Recurse into the src with a
  // local flag so the loaded value's injectivity claim is discarded.
  // Indirect-flag still propagates via out_is_indirect by reference.
  if (auto *gl = s->cast<GlobalLoadStmt>()) {
    if (inside_index && gl->src &&
        (gl->src->is<ExternalPtrStmt>() || gl->src->is<GlobalPtrStmt>() ||
         gl->src->is<MatrixPtrStmt>())) {
      out_is_indirect = true;
    }
    bool gl_inj_local = false;
    return "R(" +
           fingerprint_value(gl->src, depth + 1, gl_inj_local, out_is_indirect,
                             inside_index, load_map) +
           ")";
  }
  // ExternalPtr / GlobalPtr / MatrixPtr: the address is injective iff
  // at least one index sub-expression is independently injective. We
  // recurse into each index with a *local* flag and only set the
  // caller's flag once at the end. The base-ptr / origin sub-tree is
  // loop-invariant so its injectivity claim is discarded.
  if (auto *ep = s->cast<ExternalPtrStmt>()) {
    bool base_inj_local = false;
    std::string base_fp =
        fingerprint_value(ep->base_ptr, depth + 1, base_inj_local,
                          out_is_indirect, inside_index, load_map);
    bool any_idx_inj = false;
    std::ostringstream oss;
    oss << "EP(" << base_fp << ",[";
    for (size_t i = 0; i < ep->indices.size(); i++) {
      if (i > 0)
        oss << ",";
      bool idx_inj_local = false;
      oss << fingerprint_value(ep->indices[i], depth + 1, idx_inj_local,
                               out_is_indirect, /*inside_index=*/true,
                               load_map);
      if (idx_inj_local)
        any_idx_inj = true;
    }
    oss << "]";
    if (ep->is_grad)
      oss << "G";
    oss << ")";
    if (any_idx_inj)
      out_is_injective_in_loop_idx = true;
    return oss.str();
  }
  if (auto *gp = s->cast<GlobalPtrStmt>()) {
    bool any_idx_inj = false;
    std::ostringstream oss;
    oss << "GP(" << reinterpret_cast<uintptr_t>(gp->snode) << ",[";
    for (size_t i = 0; i < gp->indices.size(); i++) {
      if (i > 0)
        oss << ",";
      bool idx_inj_local = false;
      oss << fingerprint_value(gp->indices[i], depth + 1, idx_inj_local,
                               out_is_indirect, /*inside_index=*/true,
                               load_map);
      if (idx_inj_local)
        any_idx_inj = true;
    }
    oss << "])";
    if (any_idx_inj)
      out_is_injective_in_loop_idx = true;
    return oss.str();
  }
  if (auto *mp = s->cast<MatrixPtrStmt>()) {
    bool origin_inj_local = false;
    bool offset_inj_local = false;
    std::string o_fp =
        fingerprint_value(mp->origin, depth + 1, origin_inj_local,
                          out_is_indirect, inside_index, load_map);
    std::string off_fp =
        fingerprint_value(mp->offset, depth + 1, offset_inj_local,
                          out_is_indirect, /*inside_index=*/true, load_map);
    if (origin_inj_local || offset_inj_local)
      out_is_injective_in_loop_idx = true;
    return "MP(" + o_fp + "," + off_fp + ")";
  }
  // BinaryOp: per-operator injectivity. We recurse with local flags so
  // that, e.g., (i + 1) is correctly classified as injective while
  // (i + 1) // 2 - whose left operand recursively contains an
  // injective sub-expression - is correctly classified as NOT
  // injective.
  if (auto *bin = s->cast<BinaryOpStmt>()) {
    bool lhs_inj_local = false;
    bool rhs_inj_local = false;
    std::string lhs_fp =
        fingerprint_value(bin->lhs, depth + 1, lhs_inj_local, out_is_indirect,
                          inside_index, load_map);
    std::string rhs_fp =
        fingerprint_value(bin->rhs, depth + 1, rhs_inj_local, out_is_indirect,
                          inside_index, load_map);
    bool this_inj = false;
    switch (bin->op_type) {
      // i + c, i - c, c + i, c - i: bijection on the lane's range when
      // c is a compile-time constant (which it is for stride/offset
      // arithmetic emitted by the front-end).
      case BinaryOpType::add:
      case BinaryOpType::sub:
        this_inj = (lhs_inj_local && is_compile_time_const(bin->rhs)) ||
                   (rhs_inj_local && is_compile_time_const(bin->lhs));
        break;
      // i * c, c * i: bijection iff c is a non-zero compile-time
      // constant. c == 0 collapses every thread onto address 0; we
      // reject that case explicitly. Overflow inside int32 is not a
      // safety issue for fusion because pre- and post-fusion produce
      // the same wrapped result, but a non-injective wrap-collision
      // is impossible across a single loop's range for typical stride
      // constants - the operands here are emitted by index arithmetic
      // and would have already broken the kernel if they wrapped.
      case BinaryOpType::mul:
        this_inj = (lhs_inj_local && is_nonzero_compile_time_const(bin->rhs)) ||
                   (rhs_inj_local && is_nonzero_compile_time_const(bin->lhs));
        break;
      // i << c: equivalent to i * 2^c, bijective for any const c.
      case BinaryOpType::bit_shl:
        this_inj = lhs_inj_local && is_compile_time_const(bin->rhs);
        break;
      // i ^ c, c ^ i: XOR with a constant is a bijection on the
      // integer bit pattern. Used by some hash-style index codes.
      case BinaryOpType::bit_xor:
        this_inj = (lhs_inj_local && is_compile_time_const(bin->rhs)) ||
                   (rhs_inj_local && is_compile_time_const(bin->lhs));
        break;
      // Everything else - truediv, floordiv, mod, max, min, bit_and,
      // bit_or, bit_shr, bit_sar, pow, atan2, cmp_*, logical_*, ... -
      // is NOT provably injective in the loop index and falls into the
      // conservative "reject" bucket. This is what catches the
      // arr[i // 2] / arr[i % 2] / arr[min(i, K)] / arr[i & 1] family
      // of patterns the reviewer flagged.
      default:
        break;
    }
    if (this_inj)
      out_is_injective_in_loop_idx = true;
    return "B" + std::to_string(static_cast<int>(bin->op_type)) + "(" + lhs_fp +
           "," + rhs_fp + ")";
  }
  // UnaryOp: only the handful that are bijections preserve
  // injectivity; everything else (cast_value, sqrt, abs, sgn, sin,
  // cos, log, exp, ...) is rejected.
  if (auto *un = s->cast<UnaryOpStmt>()) {
    bool op_inj_local = false;
    std::string o_fp =
        fingerprint_value(un->operand, depth + 1, op_inj_local, out_is_indirect,
                          inside_index, load_map);
    bool this_inj = false;
    switch (un->op_type) {
      case UnaryOpType::neg:
      case UnaryOpType::bit_not:
      case UnaryOpType::cast_bits:
        this_inj = op_inj_local;
        break;
      default:
        break;
    }
    if (this_inj)
      out_is_injective_in_loop_idx = true;
    return "U" + std::to_string(static_cast<int>(un->op_type)) + "(" + o_fp +
           ")";
  }
  // Unknown / unhandled node kind: fall back to pointer identity. This
  // is conservative: identical-by-value Stmts with different pointers
  // fingerprint as unequal, so they don't match across A and B, so the
  // same-thread RAW check rejects - safe.
  std::ostringstream oss;
  oss << "@" << typeid(*s).name() << "_" << reinterpret_cast<uintptr_t>(s);
  return oss.str();
}

AccessFp make_access_fp(Stmt *ptr, const LoadMap *load_map) {
  AccessFp afp;
  afp.is_injective_in_loop_idx = false;
  afp.is_indirect = false;
  afp.fp =
      fingerprint_value(ptr, 0, afp.is_injective_in_loop_idx, afp.is_indirect,
                        /*inside_index=*/false, load_map);
  return afp;
}

// Collects the set of AllocaStmts that are written (LocalStore or
// AtomicOp on the alloca itself) inside a block subtree. Used by the
// LocalLoadResolver below to know which alloca slots become
// unresolved after a control-flow merge.
class AllocaWriteCollector : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  std::unordered_set<AllocaStmt *> written;

  void visit(LocalStoreStmt *s) override {
    if (auto *a = s->dest->cast<AllocaStmt>())
      written.insert(a);
  }
  void visit(AtomicOpStmt *s) override {
    if (auto *a = s->dest ? s->dest->cast<AllocaStmt>() : nullptr)
      written.insert(a);
  }
};

// Pre-pass that walks an offload body in execution order, tracking
// the most recent LocalStoreStmt to each AllocaStmt at each program
// point. When it sees a LocalLoadStmt, it records the stored value
// that the load would observe.
//
// IfStmt / WhileStmt handling is conservative: any alloca whose value
// might differ between branches (or has been written inside a maybe-
// run loop) becomes unresolved after the construct. We don't try to
// merge branch states - a load with potentially-different values
// across paths can't be safely fingerprinted as any single one.
class LocalLoadResolver : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  LoadMap resolved;

  // alloca -> most recent stored value at current visit position.
  // Allocas whose current value is unknown (e.g. after a conditional
  // store) are absent from this map.
  std::unordered_map<AllocaStmt *, Stmt *> cur;

  void visit(LocalStoreStmt *s) override {
    if (auto *alloca = s->dest->cast<AllocaStmt>()) {
      cur[alloca] = s->val;
    }
    // Stores via MatrixPtrStmt / GetElementStmt etc do not update
    // `cur`; the alloca's current value stays whatever it was, which
    // for matrix slot stores is "unknown" - that's the desired
    // behaviour.
  }

  void visit(AtomicOpStmt *s) override {
    if (auto *alloca = s->dest ? s->dest->cast<AllocaStmt>() : nullptr) {
      cur.erase(alloca);
    }
  }

  void visit(LocalLoadStmt *s) override {
    if (auto *alloca = s->src->cast<AllocaStmt>()) {
      auto it = cur.find(alloca);
      if (it != cur.end()) {
        resolved[s] = it->second;
      }
    }
  }

  // Drop from `cur` any alloca that is written somewhere under
  // `block`, so subsequent loads correctly see "unknown".
  void invalidate_writes_in(Block *block) {
    if (!block)
      return;
    AllocaWriteCollector wc;
    block->accept(&wc);
    for (auto *a : wc.written)
      cur.erase(a);
  }

  void visit(IfStmt *if_stmt) override {
    preprocess_container_stmt(if_stmt);
    auto saved = cur;
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    cur = saved;
    if (if_stmt->false_statements)
      if_stmt->false_statements->accept(this);
    cur = saved;
    invalidate_writes_in(if_stmt->true_statements.get());
    invalidate_writes_in(if_stmt->false_statements.get());
  }

  void visit(WhileStmt *stmt) override {
    preprocess_container_stmt(stmt);
    auto saved = cur;
    if (stmt->body)
      stmt->body->accept(this);
    cur = saved;
    invalidate_writes_in(stmt->body.get());
  }
};

// Walks a pointer-producing chain (MatrixPtrStmt -> ExternalPtrStmt ->
// ArgLoadStmt / ExternalTensorBasePtrStmt, or GlobalPtrStmt for SNode
// accesses) and returns the Resource it ultimately addresses, or kind =
// Unknown if we can't classify it.
Resource extract_resource(Stmt *ptr) {
  Resource r;
  r.kind = Resource::Kind::Unknown;
  r.ndarray_arg_id = -1;
  r.ndarray_is_grad = 0;
  r.snode = nullptr;
  r.gtmp_offset = 0;
  if (!ptr)
    return r;

  // Drill through MatrixPtrStmt (used for matrix/vector element access).
  while (auto *mp = ptr->cast<MatrixPtrStmt>()) {
    if (!mp->origin)
      return r;
    ptr = mp->origin;
  }

  if (auto *eps = ptr->cast<ExternalPtrStmt>()) {
    Stmt *base = eps->base_ptr;
    if (!base) {
      return r;
    }
    if (auto *arg = base->cast<ArgLoadStmt>()) {
      r.kind = Resource::Kind::Ndarray;
      // arg_id is a vector<int>; use the first component as the
      // identifying key. We are coarse here (we don't distinguish
      // sub-arrays of the same arg), which is intentional: anything
      // sharing the top-level arg_id is treated as the same resource.
      r.ndarray_arg_id = arg->arg_id.empty() ? -1 : arg->arg_id[0];
      r.ndarray_is_grad = eps->is_grad ? 1 : 0;
      return r;
    }
    if (auto *etbp = base->cast<ExternalTensorBasePtrStmt>()) {
      r.kind = Resource::Kind::Ndarray;
      r.ndarray_arg_id = etbp->arg_id.empty() ? -1 : etbp->arg_id[0];
      r.ndarray_is_grad = etbp->is_grad ? 1 : 0;
      return r;
    }
    return r;
  }

  if (auto *gps = ptr->cast<GlobalPtrStmt>()) {
    r.kind = Resource::Kind::SNode;
    r.snode = gps->snode;
    return r;
  }

  if (auto *gtmp = ptr->cast<GlobalTemporaryStmt>()) {
    r.kind = Resource::Kind::GlobalTmp;
    r.gtmp_offset = static_cast<int64_t>(gtmp->offset);
    return r;
  }

  // ThreadLocalPtrStmt / BlockLocalPtrStmt -> per-thread/per-block
  // scratch, never shared cross-thread, safe to ignore.
  if (ptr->is<ThreadLocalPtrStmt>() || ptr->is<BlockLocalPtrStmt>()) {
    return r;  // Kind::Unknown but caller filters these out via the
               // BasicStmtVisitor (we never call extract_resource for
               // these statements).
  }

  return r;
}

// Collects the set of Resources written and read inside an
// OffloadedStmt's body. has_unknown is set if we encounter a
// global-memory operation whose target we can't classify; in that case
// the caller treats the whole offload as touching "everything" and
// refuses to fuse it.
//
// For each resource we also record the set of address fingerprints
// observed at every write / read of that resource. This is consumed
// by the same-thread RAW check (QD_AGGRESSIVE_KERNEL_FUSION_RAW=1) to prove
// that each byte touched by both A and B is only touched by the same thread
// index in both bodies.
class CollectGlobalAccesses : public BasicStmtVisitor {
 public:
  using FpSet = std::unordered_set<AccessFp, AccessFpHash>;
  std::unordered_map<Resource, FpSet, ResourceHash> writes;
  std::unordered_map<Resource, FpSet, ResourceHash> reads;
  bool has_unknown = false;
  // Optional. When set, fingerprinting will see through
  // LocalLoad(alloca) by looking up the load's resolved value
  // (the most recent LocalStore in body order).
  const LoadMap *load_map = nullptr;

  using BasicStmtVisitor::visit;

  bool touches_write(const Resource &r) const {
    return writes.find(r) != writes.end();
  }
  bool touches_read(const Resource &r) const {
    return reads.find(r) != reads.end();
  }

  void record(Stmt *ptr, bool is_write, bool is_read) {
    if (ptr &&
        (ptr->is<ThreadLocalPtrStmt>() || ptr->is<BlockLocalPtrStmt>())) {
      return;
    }
    auto r = extract_resource(ptr);
    if (r.kind == Resource::Kind::Unknown) {
      has_unknown = true;
      return;
    }
    AccessFp afp = make_access_fp(ptr, load_map);
    if (is_write)
      writes[r].insert(afp);
    if (is_read)
      reads[r].insert(afp);
  }

  void visit(GlobalStoreStmt *stmt) override {
    record(stmt->dest, /*is_write=*/true, /*is_read=*/false);
  }

  void visit(GlobalLoadStmt *stmt) override {
    record(stmt->src, /*is_write=*/false, /*is_read=*/true);
  }

  void visit(AtomicOpStmt *stmt) override {
    // Atomic op reads+writes the same destination.
    record(stmt->dest, /*is_write=*/true, /*is_read=*/true);
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    // Conservative: a foreign call can touch anything.
    has_unknown = true;
  }

  void visit(InternalFuncStmt *stmt) override {
    // Quadrants intrinsics: most are pure (e.g. shfl, ballot, math),
    // but a few have memory side effects. Be conservative.
    has_unknown = true;
  }

  void visit(ClearListStmt *stmt) override {
    // Touches snode meta. Don't fuse across.
    has_unknown = true;
  }

  void visit(RandStmt *stmt) override {
    // Each invocation advances the runtime random state. Reordering
    // two RandStmts changes which thread gets which random number,
    // which can perturb deterministic physics. Treat as unknown.
    has_unknown = true;
  }

  void visit(SNodeOpStmt *stmt) override {
    // Allocation / activation / deactivation of snode cells touches
    // snode metadata. Conservative.
    has_unknown = true;
  }

  void visit(AssertStmt *stmt) override {
    // Asserts have observable behavior (program termination on
    // failure). Don't reorder across them.
    has_unknown = true;
  }
};

// Coarse compile-config knob: *disabled by default*. Set
// QD_AGGRESSIVE_KERNEL_FUSION to a truthy value (1 / on / true / yes)
// to enable the pass; default behaviour is the historical "no fusion"
// pipeline. The pass is opt-in because a) it has no register-pressure
// / LDS / I-cache cost model so it can regress occupancy-bound kernels
// after fusion, and b) the same-thread RAW relaxation is a syntactic
// safety check, not a real loop-dependence analysis. Workloads that
// have been validated under fusion (currently: Genesis on AMDGPU) can
// opt in via this env var.
bool fuse_enabled() {
  static const bool enabled = []() {
    const char *flag = std::getenv("QD_AGGRESSIVE_KERNEL_FUSION");
    if (!flag || flag[0] == '\0')
      return false;
    std::string s(flag);
    for (auto &c : s)
      c = (char)std::tolower((unsigned char)c);
    if (s == "0" || s == "off" || s == "false" || s == "no")
      return false;
    return true;
  }();
  return enabled;
}

bool diag_enabled() {
  static const bool enabled = []() {
    const char *flag = std::getenv("QD_AGGRESSIVE_KERNEL_FUSION_DIAG");
    return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
  }();
  return enabled;
}

// Treats a resource shared between A and B as safe to fuse provided
// every access (write or read) to that resource in either body uses
// the *same* per-thread, non-indirect address fingerprint. This
// unlocks the common Genesis pattern
//   arr[i] = ...   (offload A)
//   ...    = arr[i] (offload B)
// where the kernel boundary between the writer and reader doesn't
// serve any cross-thread purpose: thread T writes arr[T] and thread T
// reads arr[T], and no other thread touches arr[T] post-fusion.
//
// Defaults to on when fusion is enabled because it gave a consistent
// throughput lift in benchmarking. Set QD_AGGRESSIVE_KERNEL_FUSION_RAW=0
// to disable (kill switch); the resource-disjoint policy still applies.
bool raw_enabled() {
  static const bool enabled = []() {
    const char *flag = std::getenv("QD_AGGRESSIVE_KERNEL_FUSION_RAW");
    if (!flag || flag[0] == '\0')
      return true;
    std::string s = flag;
    for (auto &c : s)
      c = (char)std::tolower((unsigned char)c);
    if (s == "0" || s == "off" || s == "false" || s == "no")
      return false;
    return true;
  }();
  return enabled;
}

// True iff a and b have identical, statically-comparable launch
// dimensions. Conservative: only accept compile-time-known bounds.
// Dynamic-bound fusion is handled separately in can_fuse_with_reason
// (after we have the access analysis) so we can prove the bound
// gtmp slot isn't written by A's body.
bool same_static_launch_dims(const OffloadedStmt *a, const OffloadedStmt *b) {
  if (!a->const_begin || !a->const_end)
    return false;
  if (!b->const_begin || !b->const_end)
    return false;
  if (a->begin_value != b->begin_value)
    return false;
  if (a->end_value != b->end_value)
    return false;
  if (a->block_dim != b->block_dim)
    return false;
  return true;
}

// True iff a and b have identical dynamic-via-gtmp bounds (no
// end_stmt, both read the same gtmp slot for their end). Caller must
// additionally verify no statement in A writes to that gtmp slot.
bool same_gtmp_launch_dims(const OffloadedStmt *a, const OffloadedStmt *b) {
  // Bounds must match in kind (both static or both gtmp-dynamic) on
  // both sides.
  if (a->const_begin != b->const_begin)
    return false;
  if (a->const_end != b->const_end)
    return false;
  // We only handle the dynamic-via-gtmp case here.
  if (a->const_end)
    return false;  // Handled by same_static_launch_dims.
  if (a->end_stmt != nullptr || b->end_stmt != nullptr)
    return false;
  if (a->const_begin) {
    if (a->begin_value != b->begin_value)
      return false;
  } else {
    if (a->begin_offset != b->begin_offset)
      return false;
  }
  if (a->end_offset != b->end_offset)
    return false;
  if (a->block_dim != b->block_dim)
    return false;
  return true;
}

enum class FuseReject : int {
  Ok = 0,
  NotRangeFor,
  TlsBls,
  MemAccessOpt,
  GridDim,
  Bounds,
  UnknownAccess,
  RaceWriteWrite,
  RaceWriteRead,
  RaceReadWrite,
  DynamicBoundRace,
};

const char *reject_name(FuseReject r) {
  switch (r) {
    case FuseReject::Ok:
      return "ok";
    case FuseReject::NotRangeFor:
      return "not-range-for";
    case FuseReject::TlsBls:
      return "tls-or-bls-prologue";
    case FuseReject::MemAccessOpt:
      return "mem-access-opt";
    case FuseReject::GridDim:
      return "grid-dim-mismatch";
    case FuseReject::Bounds:
      return "bounds-mismatch-or-dynamic";
    case FuseReject::UnknownAccess:
      return "unknown-access";
    case FuseReject::RaceWriteWrite:
      return "race-write-write";
    case FuseReject::RaceWriteRead:
      return "race-a-read-b-write";
    case FuseReject::RaceReadWrite:
      return "race-a-write-b-read";
    case FuseReject::DynamicBoundRace:
      return "dynamic-bound-write-race";
  }
  return "?";
}

FuseReject can_fuse_with_reason(OffloadedStmt *a, OffloadedStmt *b) {
  using Type = OffloadedStmt::TaskType;
  if (a->task_type != Type::range_for || b->task_type != Type::range_for)
    return FuseReject::NotRangeFor;
  if (a->tls_prologue || a->tls_epilogue || b->tls_prologue || b->tls_epilogue)
    return FuseReject::TlsBls;
  if (a->bls_prologue || a->bls_epilogue || b->bls_prologue || b->bls_epilogue)
    return FuseReject::TlsBls;
  if (!a->mem_access_opt.get_all().empty() ||
      !b->mem_access_opt.get_all().empty())
    return FuseReject::MemAccessOpt;
  if (a->grid_dim != b->grid_dim)
    return FuseReject::GridDim;
  const bool static_dims = same_static_launch_dims(a, b);
  // Dynamic-via-gtmp matching is on by default. It is only valid
  // because this pass is sequenced *after* FixCrossOffloadReferences,
  // which is where OffloadedStmt::end_offset / begin_offset get
  // populated with their final gtmp byte offsets. Run earlier and
  // these fields are still their default value (0) for every
  // dynamic-bound offload, causing the matcher to incorrectly accept
  // unrelated offloads as "same bound" and produce GPU memory faults.
  const bool gtmp_dims = !static_dims && same_gtmp_launch_dims(a, b);
  if (!static_dims && !gtmp_dims)
    return FuseReject::Bounds;

  // Pre-pass: walk each body in execution order and resolve every
  // LocalLoadStmt to the most recent LocalStore's value. This lets
  // the fingerprinter see through `LocalLoad(alloca)` to the
  // expression that originally produced the stored value (typically
  // a LoopIndexStmt or arithmetic on one). Without this every
  // physics-body access fingerprints through an opaque LocalLoadStmt
  // pointer and the same-thread RAW relaxation never triggers.
  LocalLoadResolver a_loads, b_loads;
  a->body->accept(&a_loads);
  b->body->accept(&b_loads);

  CollectGlobalAccesses a_acc, b_acc;
  a_acc.load_map = &a_loads.resolved;
  b_acc.load_map = &b_loads.resolved;
  a->body->accept(&a_acc);
  b->body->accept(&b_acc);
  if (a_acc.has_unknown || b_acc.has_unknown)
    return FuseReject::UnknownAccess;

  // Dynamic-bound case: pre-fusion the host evaluates A's bound, runs
  // A, then re-evaluates B's bound and runs B. Post-fusion the host
  // evaluates the bound *once* and uses it for both bodies. If A's
  // body writes to the gtmp slot from which B reads its bound, B's
  // iteration count differs across pre/post fusion - unsafe.
  if (gtmp_dims) {
    Resource gtmp_end;
    gtmp_end.kind = Resource::Kind::GlobalTmp;
    gtmp_end.gtmp_offset = static_cast<int64_t>(a->end_offset);
    gtmp_end.ndarray_arg_id = -1;
    gtmp_end.ndarray_is_grad = 0;
    gtmp_end.snode = nullptr;
    if (a_acc.writes.count(gtmp_end))
      return FuseReject::DynamicBoundRace;
    if (!a->const_begin) {
      Resource gtmp_begin = gtmp_end;
      gtmp_begin.gtmp_offset = static_cast<int64_t>(a->begin_offset);
      if (a_acc.writes.count(gtmp_begin))
        return FuseReject::DynamicBoundRace;
    }
  }

  // Per-resource conflict check. A pair (A,B) is racy on a resource
  // iff that resource is touched by both bodies *and* at least one
  // body writes to it. Under the default policy we reject any such
  // conflict. Under same-thread RAW relaxation
  // (QD_AGGRESSIVE_KERNEL_FUSION_RAW=1) we instead check whether every access
  // (write or read) to the shared resource in either body has the same
  // per-thread address fingerprint, in which case thread T touches the same
  // byte in both A and B and no other thread races against it.
  const bool raw = raw_enabled();
  std::unordered_set<Resource, ResourceHash> all_resources;
  for (const auto &kv : a_acc.writes)
    all_resources.insert(kv.first);
  for (const auto &kv : a_acc.reads)
    all_resources.insert(kv.first);
  for (const auto &kv : b_acc.writes)
    all_resources.insert(kv.first);
  for (const auto &kv : b_acc.reads)
    all_resources.insert(kv.first);

  for (const auto &r : all_resources) {
    const bool a_w = a_acc.writes.count(r) > 0;
    const bool a_r = a_acc.reads.count(r) > 0;
    const bool b_w = b_acc.writes.count(r) > 0;
    const bool b_r = b_acc.reads.count(r) > 0;
    if (!(a_w || a_r))
      continue;  // only B touches r
    if (!(b_w || b_r))
      continue;  // only A touches r
    if (!a_w && !b_w)
      continue;  // both sides read; no race

    FuseReject reason;
    if (a_w && b_w)
      reason = FuseReject::RaceWriteWrite;
    else if (a_r && b_w)
      reason = FuseReject::RaceReadWrite;
    else
      reason = FuseReject::RaceWriteRead;

    if (!raw) {
      return reason;
    }

    // Collect every (write+read) address fingerprint on this
    // resource across both bodies. Safe iff there is exactly one,
    // and it is per-thread.
    std::unordered_set<AccessFp, AccessFpHash> all_fps;
    auto add_fps =
        [&](const std::unordered_map<Resource, CollectGlobalAccesses::FpSet,
                                     ResourceHash> &m) {
          auto it = m.find(r);
          if (it == m.end())
            return;
          for (const auto &fp : it->second)
            all_fps.insert(fp);
        };
    add_fps(a_acc.writes);
    add_fps(a_acc.reads);
    add_fps(b_acc.writes);
    add_fps(b_acc.reads);
    bool admit = all_fps.size() == 1 &&
                 all_fps.begin()->is_injective_in_loop_idx &&
                 !all_fps.begin()->is_indirect;
    if (!admit) {
      return reason;
    }
    // Single non-indirect per-thread fingerprint shared across A and
    // B: thread T's access touches the same byte in both bodies, and
    // no other thread touches that byte post-fusion. Safe.
  }
  return FuseReject::Ok;
}

void merge_b_into_a(OffloadedStmt *a, OffloadedStmt *b) {
  // Rewire any reference to B (e.g. LoopIndexStmt's whose .loop ==
  // B) so they point at A. We walk *both* offload bodies because B's
  // body may contain LoopIndexStmt(loop=B,...), and A's body must not
  // gain dangling references to B once we drop it. Mirrors the
  // pattern used in Offloader::run when a RangeForStmt is replaced by
  // its containing OffloadedStmt.
  irpass::replace_all_usages_with(a, b, a);
  irpass::replace_all_usages_with(b, b, a);

  // Move B's body statements into A's body. Each statement's parent
  // block pointer is updated by Block::insert.
  for (size_t j = 0; j < b->body->statements.size(); j++) {
    a->body->insert(std::move(b->body->statements[j]));
  }
  b->body->statements.clear();
}

int fuse_pass(Block *root_block) {
  int fused_count = 0;
  bool changed = true;
  // Track per-reason reject counts for the diagnostic dump.
  std::unordered_map<int, int> reject_counts;
  while (changed) {
    changed = false;
    auto &stmts = root_block->statements;
    for (size_t i = 0; i + 1 < stmts.size();) {
      auto *a = stmts[i]->cast<OffloadedStmt>();
      auto *b = stmts[i + 1]->cast<OffloadedStmt>();
      FuseReject reason =
          (a && b) ? can_fuse_with_reason(a, b) : FuseReject::NotRangeFor;
      if (reason == FuseReject::Ok) {
        if (diag_enabled()) {
          fmt::print(
              stderr,
              "[fuse_offloaded_tasks] fusing offloads at indices {} and {} "
              "(begin={}, end={}, block_dim={}, const_begin={}, "
              "const_end={}, begin_offset={}, end_offset={})\n",
              i, i + 1, a->begin_value, a->end_value, a->block_dim,
              a->const_begin, a->const_end, a->begin_offset, a->end_offset);
        }
        merge_b_into_a(a, b);
        stmts.erase(stmts.begin() + (long)i + 1);
        fused_count++;
        changed = true;
        // Don't advance i: try fusing the new neighbor.
      } else {
        reject_counts[(int)reason]++;
        i++;
      }
    }
  }
  if (diag_enabled() && !reject_counts.empty()) {
    fmt::print(stderr, "[fuse_offloaded_tasks] reject reasons: ");
    for (const auto &kv : reject_counts) {
      fmt::print(stderr, "{}={} ", reject_name((FuseReject)kv.first),
                 kv.second);
    }
    fmt::print(stderr, "\n");
  }
  return fused_count;
}

}  // namespace

namespace irpass {

void fuse_offloaded_tasks(IRNode *root) {
  QD_AUTO_PROF;
  if (!fuse_enabled())
    return;
  auto *root_block = root->cast<Block>();
  if (!root_block)
    return;
  // Opt out of fusion entirely when the user explicitly asked for the
  // cuda_graph dispatch model. The cuda_graph manager relies on each
  // top-level for-loop producing its own OffloadedStmt so it can
  // capture them as graph nodes; collapsing them here would silently
  // change the dispatch shape the user requested. The flag is
  // mirrored from Python onto Kernel::use_cuda_graph at materialize
  // time, so we just inspect the first offload's owning kernel.
  for (auto &s : root_block->statements) {
    if (auto *o = s->cast<OffloadedStmt>()) {
      if (o->kernel_ && o->kernel_->use_cuda_graph)
        return;
      break;
    }
  }
  // Diagnostic: print how many top-level offloads this kernel has and
  // how many adjacent (A,B) pairs we considered, so we can tell
  // whether (a) we even saw candidates and (b) which fusion check
  // rejected them.
  int n_offloads = 0;
  int n_range_for = 0;
  for (auto &s : root_block->statements) {
    if (auto *o = s->cast<OffloadedStmt>()) {
      n_offloads++;
      if (o->task_type == OffloadedStmt::TaskType::range_for)
        n_range_for++;
    }
  }
  int n = fuse_pass(root_block);
  if (diag_enabled()) {
    fmt::print(stderr,
               "[fuse_offloaded_tasks] kernel: {} top-level offloads, {} "
               "range_for, fused {} pair(s) (raw={})\n",
               n_offloads, n_range_for, n, raw_enabled() ? "on" : "off");
  }
}

}  // namespace irpass

}  // namespace quadrants::lang
