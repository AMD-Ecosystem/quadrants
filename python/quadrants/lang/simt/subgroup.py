# type: ignore
"""Wave-scope (a.k.a. SPIR-V "Subgroup") primitives.

Backend availability for the wave-scope reductions added in this module:

- AMDGPU: full implementation via ``patch_intrinsic`` (see
  ``quadrants/runtime/llvm/llvm_context.cpp``); ``reduce_or_i32`` /
  ``reduce_and_i32`` lower to ``amdgcn.icmp`` + ``amdgcn.ballot``
  (single-instruction-equivalent at AMDGCN level), and ``barrier`` lowers
  to ``llvm.amdgcn.wave.barrier`` (compiler + memory-ordering barrier
  scoped to the current wave; no hardware sync, no deadlock hazard on
  partial-wave participation).
- Other LLVM-runtime backends (CPU, CUDA today): no patcher yet —
  ``reduce_or_i32`` / ``reduce_and_i32`` fall through to the runtime stubs
  in ``quadrants/runtime/llvm/runtime_module/runtime.cpp`` which return 0.
  This is intentional placeholder behaviour for backends where wave-scope
  semantics aren't wired up; callers that need cross-backend portability
  should gate on ``qd.cfg.arch == qd.amdgpu`` for now.
- SPIR-V backends (Vulkan, Metal): the polymorphic ``reduce_or`` /
  ``reduce_and`` (no ``_i32`` suffix) cover those targets.
"""

from quadrants.lang import impl


def barrier():
    """Wave-scope (subgroup) execution + memory-ordering barrier.

    On AMDGPU lowers to ``llvm.amdgcn.wave.barrier``: no hardware wait
    (all 64 lanes of an AMDGPU wave already execute in lockstep) but
    prevents the compiler from reordering memory ops or moving
    convergent ops across this point. Safe to call from non-uniform
    control flow — does NOT synchronize across waves in a workgroup.
    For workgroup-scope synchronization use ``qd.simt.block.sync()``.
    """
    return impl.call_internal("subgroupBarrier", with_runtime_context=False)


def memory_barrier():
    return impl.call_internal("subgroupMemoryBarrier", with_runtime_context=False)


def elect():
    return impl.call_internal("subgroupElect", with_runtime_context=False)


def all_true(cond):
    # TODO
    pass


def any_true(cond):
    # TODO
    pass


def reduce_or_i32(value):
    """Wave-scope OR reduction (boolean): returns 1 if any active lane
    in the wave has value != 0, else 0.

    AMDGPU implementation: ``s_or_b64``-equivalent via ``amdgcn.icmp.i32``
    ballot, single-instruction at the AMDGCN level after LLVM lowering.
    See ``quadrants/runtime/llvm/llvm_context.cpp`` for the patcher.

    Differs from the polymorphic ``reduce_or(value)`` (SPIRV-only at the
    moment): this variant is i32-typed and treats input as boolean
    (0 vs non-zero), sufficient for wave-vote convergence checks.
    """
    return impl.call_internal("subgroupOr_i32", value, with_runtime_context=False)


def reduce_and_i32(value):
    """Wave-scope AND reduction (boolean): returns 1 if all active lanes
    in the wave have value != 0, else 0. See ``reduce_or_i32`` for context.
    """
    return impl.call_internal("subgroupAnd_i32", value, with_runtime_context=False)


def all_equal(value):
    # TODO
    pass


def broadcast_first(value):
    # TODO
    pass


def broadcast(value, index):
    return impl.call_internal("subgroupBroadcast", value, index, with_runtime_context=False)


def group_size():
    return impl.call_internal("subgroupSize", with_runtime_context=False)


def invocation_id():
    return impl.call_internal("subgroupInvocationId", with_runtime_context=False)


def reduce_add(value):
    return impl.call_internal("subgroupAdd", value, with_runtime_context=False)


def reduce_mul(value):
    return impl.call_internal("subgroupMul", value, with_runtime_context=False)


def reduce_min(value):
    return impl.call_internal("subgroupMin", value, with_runtime_context=False)


def reduce_max(value):
    return impl.call_internal("subgroupMax", value, with_runtime_context=False)


def reduce_and(value):
    return impl.call_internal("subgroupAnd", value, with_runtime_context=False)


def reduce_or(value):
    return impl.call_internal("subgroupOr", value, with_runtime_context=False)


def reduce_xor(value):
    return impl.call_internal("subgroupXor", value, with_runtime_context=False)


def inclusive_add(value):
    return impl.call_internal("subgroupInclusiveAdd", value, with_runtime_context=False)


def inclusive_mul(value):
    return impl.call_internal("subgroupInclusiveMul", value, with_runtime_context=False)


def inclusive_min(value):
    return impl.call_internal("subgroupInclusiveMin", value, with_runtime_context=False)


def inclusive_max(value):
    return impl.call_internal("subgroupInclusiveMax", value, with_runtime_context=False)


def inclusive_and(value):
    return impl.call_internal("subgroupInclusiveAnd", value, with_runtime_context=False)


def inclusive_or(value):
    return impl.call_internal("subgroupInclusiveOr", value, with_runtime_context=False)


def inclusive_xor(value):
    return impl.call_internal("subgroupInclusiveXor", value, with_runtime_context=False)


def exclusive_add(value):
    # TODO
    pass


def exclusive_mul(value):
    # TODO
    pass


def exclusive_min(value):
    # TODO
    pass


def exclusive_max(value):
    # TODO
    pass


def exclusive_and(value):
    # TODO
    pass


def exclusive_or(value):
    # TODO
    pass


def exclusive_xor(value):
    # TODO
    pass


def shuffle(value, index):
    return impl.call_internal("subgroupShuffle", value, index, with_runtime_context=False)


def shuffle_xor(value, mask):
    # TODO
    pass


def shuffle_up(value, offset):
    return impl.call_internal("subgroupShuffleUp", value, offset, with_runtime_context=False)


def shuffle_down(value, offset):
    return impl.call_internal("subgroupShuffleDown", value, offset, with_runtime_context=False)


__all__ = [
    "barrier",
    "memory_barrier",
    "elect",
    "all_true",
    "any_true",
    "all_equal",
    "broadcast_first",
    "reduce_add",
    "reduce_mul",
    "reduce_min",
    "reduce_max",
    "reduce_and",
    "reduce_and_i32",
    "reduce_or",
    "reduce_or_i32",
    "reduce_xor",
    "inclusive_add",
    "inclusive_mul",
    "inclusive_min",
    "inclusive_max",
    "inclusive_and",
    "inclusive_or",
    "inclusive_xor",
    "exclusive_add",
    "exclusive_mul",
    "exclusive_min",
    "exclusive_max",
    "exclusive_and",
    "exclusive_or",
    "exclusive_xor",
    "shuffle",
    "shuffle_xor",
    "shuffle_up",
    "shuffle_down",
]
