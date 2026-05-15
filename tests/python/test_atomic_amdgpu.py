"""AMDGPU-targeted atomic correctness tests.

These tests exist as a regression net for the atomic-ordering relaxation
landed in PR #38 (Monotonic + ``syncscope("agent")`` instead of
SequentiallyConsistent + system scope) and the follow-up correctness fixes
planned in Phase 3 (mixed-scope cleanup, CAS-loop atomic load).

Coverage is intentionally broader than ``test_atomic.py`` for the AMDGPU
arch specifically, because several existing tests are explicitly gated to
``[qd.cpu, qd.cuda]`` and therefore would not catch AMDGPU-side
regressions. Anything that exists for CPU/CUDA and could plausibly behave
differently on AMDGPU under relaxed atomics gets mirrored here.

Each test runs *only* on AMDGPU and is skipped (rather than xfailed) on
other arches so we never accidentally validate the AMDGPU contract using
a CUDA / CPU run.
"""

import pytest

import quadrants as qd

from tests import test_utils

# Keep this small enough to run quickly on CI but large enough that the
# atomics actually contend (i.e. multiple threads hit the same counter).
N = 256


# ---------------------------------------------------------------------------
# 1. End-to-end correctness for every (type, op) the AMDGPU codegen
#    rewrites with the new ordering. Mirrors the CPU/CUDA-only tests in
#    test_atomic.py.
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_add_i32_amdgpu():
    c = qd.field(qd.i32, shape=())
    c[None] = 0

    @qd.kernel
    def k():
        for _ in range(N):
            qd.atomic_add(c[None], 1)

    k()
    assert c[None] == N


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_add_i64_amdgpu():
    c = qd.field(qd.i64, shape=())
    c[None] = 0

    @qd.kernel
    def k():
        for _ in range(N):
            qd.atomic_add(c[None], qd.cast(1, qd.i64))

    k()
    assert c[None] == N


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_add_u32_amdgpu():
    c = qd.field(qd.u32, shape=())
    c[None] = 0

    @qd.kernel
    def k():
        for _ in range(N):
            qd.atomic_add(c[None], qd.cast(1, qd.u32))

    k()
    assert c[None] == N


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_add_u64_amdgpu():
    c = qd.field(qd.u64, shape=())
    c[None] = 0

    @qd.kernel
    def k():
        for _ in range(N):
            qd.atomic_add(c[None], qd.cast(1, qd.u64))

    k()
    assert c[None] == N


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_add_f32_amdgpu():
    c = qd.field(qd.f32, shape=())
    c[None] = 0.0

    @qd.kernel
    def k():
        for _ in range(N):
            qd.atomic_add(c[None], 1.0)

    k()
    # Reduction over many threads; allow a small relative tolerance for
    # ordering-dependent FP non-associativity.
    assert c[None] == test_utils.approx(float(N), rel=1e-5)


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_add_f64_amdgpu():
    c = qd.field(qd.f64, shape=())
    c[None] = 0.0

    @qd.kernel
    def k():
        for _ in range(N):
            qd.atomic_add(c[None], 1.0)

    k()
    assert c[None] == test_utils.approx(float(N), rel=1e-10)


# ---------------------------------------------------------------------------
# Integer min/max — this is the gap left by test_atomic_min_max_uint, which
# is gated to [cpu, cuda]. AMDGPU now exercises the same path under
# Monotonic+agent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u32, qd.u64])
@test_utils.test(arch=[qd.amdgpu])
def test_atomic_min_int_amdgpu(dtype):
    x = qd.field(dtype, shape=N)

    @qd.kernel
    def init():
        # x[1..N-1] = 1, 2, ..., N-1. Then x[0] is overwritten to a
        # sentinel so the reduction below has to actually update it.
        # The minimum value present in the indices the reducer scans
        # (1..N-1) is 1, which is what the assertion below checks.
        for i in x:
            x[i] = qd.cast(i, dtype)
        x[0] = qd.cast(10**6, dtype)

    @qd.kernel
    def reduce_min():
        for i in range(1, N):
            qd.atomic_min(x[0], x[i])

    init()
    reduce_min()
    assert int(x[0]) == 1


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u32, qd.u64])
@test_utils.test(arch=[qd.amdgpu])
def test_atomic_max_int_amdgpu(dtype):
    x = qd.field(dtype, shape=N)

    @qd.kernel
    def init():
        for i in x:
            x[i] = qd.cast(i + 1, dtype)

    @qd.kernel
    def reduce_max():
        for i in range(N):
            qd.atomic_max(x[0], x[i])

    init()
    reduce_max()
    assert int(x[0]) == N


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u32, qd.u64])
@test_utils.test(arch=[qd.amdgpu])
def test_atomic_bitops_int_amdgpu(dtype):
    # and / or / xor on the same destination, mirroring the
    # *_expr_evaled tests in test_atomic.py but on AMDGPU specifically.
    #
    # Host-scope assignments (``val[None] = ...``) take a Python int and
    # let the field setter do the dtype conversion. Calling ``qd.cast``
    # from host scope returns an ``Expr`` whose construction reads
    # ``src_info_stack[-1]``, which is empty outside a kernel.
    n_bits = 16
    val = qd.field(dtype, shape=())
    val[None] = (1 << n_bits) - 1

    @qd.kernel
    def do_and():
        for i in range(n_bits):
            # Clear bit i.
            qd.atomic_and(val[None], qd.cast(~(1 << i), dtype))

    do_and()
    assert int(val[None]) == 0

    val[None] = 0

    @qd.kernel
    def do_or():
        for i in range(n_bits):
            qd.atomic_or(val[None], qd.cast(1 << i, dtype))

    do_or()
    assert int(val[None]) == (1 << n_bits) - 1

    val[None] = 0

    @qd.kernel
    def do_xor():
        for i in range(n_bits):
            qd.atomic_xor(val[None], qd.cast(1 << i, dtype))

    do_xor()
    assert int(val[None]) == (1 << n_bits) - 1


# ---------------------------------------------------------------------------
# Float min/max — exercises the runtime-helper fall-through in
# real_type_atomic for f32/f64. This is the call site that is currently
# (pre-Phase-3) mixed-scope: the runtime helper uses SeqCst+system while
# atomic_add on the same dest uses Monotonic+agent. The end-to-end
# correctness still holds for these isolated reductions, so the tests
# below are expected to pass today; the mixed-scope IR-level concern is
# asserted in test_atomic_amdgpu_ir.py.
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_min_f32_amdgpu():
    x = qd.field(qd.f32, shape=N)

    @qd.kernel
    def init():
        # See test_atomic_min_int_amdgpu for why we use ``i`` (not
        # ``i + 1``): the reducer scans x[1..N-1], whose smallest value
        # must equal the assertion below.
        for i in x:
            x[i] = qd.cast(i, qd.f32)
        x[0] = 1e9

    @qd.kernel
    def reduce_min():
        for i in range(1, N):
            qd.atomic_min(x[0], x[i])

    init()
    reduce_min()
    assert x[0] == test_utils.approx(1.0)


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_max_f32_amdgpu():
    x = qd.field(qd.f32, shape=N)

    @qd.kernel
    def init():
        for i in x:
            x[i] = qd.cast(i + 1, qd.f32)

    @qd.kernel
    def reduce_max():
        for i in range(N):
            qd.atomic_max(x[0], x[i])

    init()
    reduce_max()
    assert x[0] == test_utils.approx(float(N))


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_min_f64_amdgpu():
    x = qd.field(qd.f64, shape=N)

    @qd.kernel
    def init():
        # See test_atomic_min_int_amdgpu for why we use ``i`` (not
        # ``i + 1``): the reducer scans x[1..N-1], whose smallest value
        # must equal the assertion below.
        for i in x:
            x[i] = qd.cast(i, qd.f64)
        x[0] = 1e18

    @qd.kernel
    def reduce_min():
        for i in range(1, N):
            qd.atomic_min(x[0], x[i])

    init()
    reduce_min()
    assert x[0] == test_utils.approx(1.0, rel=1e-12)


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_max_f64_amdgpu():
    x = qd.field(qd.f64, shape=N)

    @qd.kernel
    def init():
        for i in x:
            x[i] = qd.cast(i + 1, qd.f64)

    @qd.kernel
    def reduce_max():
        for i in range(N):
            qd.atomic_max(x[0], x[i])

    init()
    reduce_max()
    assert x[0] == test_utils.approx(float(N), rel=1e-12)


# ---------------------------------------------------------------------------
# atomic_mul — both integer and float fall through to
# atomic_op_using_cas, which the PR explicitly overrides. Make sure the
# CAS loop is still correct under Monotonic+agent.
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_mul_i32_amdgpu():
    # Single-thread mul; multi-thread mul is sensitive to non-associativity
    # in floats, but for ints it should produce a deterministic result.
    val = qd.field(qd.i32, shape=())
    val[None] = 1

    @qd.kernel
    def k():
        for i in range(1, 8):  # serial loop -> single thread / deterministic
            qd.atomic_mul(val[None], i)

    k()
    assert int(val[None]) == 5040


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_mul_f32_amdgpu():
    val = qd.field(qd.f32, shape=())
    val[None] = 1.0

    @qd.kernel
    def k():
        for i in range(1, 8):
            qd.atomic_mul(val[None], qd.cast(i, qd.f32))

    k()
    assert val[None] == test_utils.approx(5040.0)


# ---------------------------------------------------------------------------
# 2. Index-allocator pattern (the central "is the relaxation safe?" test).
#
# Pattern: many threads atomically increment a counter; each thread writes
# its value to data[old_counter]. A *second* kernel reads counter and
# data[]. The kernel-launch boundary supplies the acquire/release ordering
# this PR's design depends on, so this should pass under Monotonic+agent.
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_index_allocator_inter_kernel_amdgpu():
    counter = qd.field(qd.i32, shape=())
    data = qd.field(qd.i32, shape=N)
    counter[None] = 0

    @qd.kernel
    def producer():
        for i in range(N):
            idx = qd.atomic_add(counter[None], 1)
            data[idx] = i + 1  # tag with a non-zero value

    @qd.kernel
    def consumer(out: qd.types.ndarray(qd.i32, 1)):
        for i in range(N):
            out[i] = data[i]

    producer()
    import numpy as np

    out = np.zeros(N, dtype=np.int32)
    consumer(out)

    assert int(counter[None]) == N
    # Every slot should have been written by exactly one producer thread,
    # so the set of values written must be exactly {1, ..., N}.
    assert sorted(out.tolist()) == list(range(1, N + 1))


# ---------------------------------------------------------------------------
# 3. Intra-kernel publish/subscribe — DOCUMENTED LIMITATION.
#
# Under the PR's Monotonic+agent ordering, a non-atomic write that follows
# an atomic on a different address has NO ordering guarantee to other
# threads in the same kernel. We do not assert correctness for this case;
# the test exists to mark the contract change explicitly and will be
# wired up if/when Phase 3.x adds an opt-in stronger fence primitive.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Intra-kernel atomic-publish-then-non-atomic-write is not "
        "supported on AMDGPU under the Monotonic+agent ordering chosen "
        "in PR #38. See docs/source/user_guide/amdgpu_atomics.md."
    )
)
@test_utils.test(arch=[qd.amdgpu])
def test_atomic_intra_kernel_publish_unsupported_amdgpu():
    # Left here as a placeholder so the contract change is greppable.
    pass


# ---------------------------------------------------------------------------
# 4. atomic_add on a global counter under heavy contention.
#
# This is the dominant pattern named in PR #38 (Genesis work-queue
# counters in narrowphase / broadphase / inequality constraint kernels).
# Test it explicitly so that any future regression in the contended path
# shows up loudly.
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.amdgpu])
def test_atomic_add_high_contention_amdgpu():
    counter = qd.field(qd.i32, shape=())
    counter[None] = 0

    big_n = 4096

    @qd.kernel
    def k():
        for _ in range(big_n):
            qd.atomic_add(counter[None], 1)

    k()
    assert int(counter[None]) == big_n
