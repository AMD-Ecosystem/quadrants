"""Tests for the same-thread RAW relaxation in fuse_offloaded_tasks.

Two adjacent offload kernels A then B that touch the same resource can
be fused into one OffloadedStmt iff every access to the resource in
either body uses the same per-thread address AND that address is a
provably *injective* function of the loop index. The injectivity check
lives in `quadrants/transforms/fuse_offloaded_tasks.cpp::fingerprint_value`.

Two failure modes the fingerprint must guard against:

1. Injective patterns over-rejected. arr[i], arr[2 * i + 3], arr[i ^ 1]
   etc. are bijective in i, so the fused output must match the un-fused
   reference byte-for-byte (every thread touches a unique byte).

2. Non-injective patterns wrongly admitted. arr[i // 2], arr[i % 4],
   arr[i & 1], arr[min(i, K)], arr[0] all collapse multiple loop
   iterations onto the same byte. Pre-fusion the kernel boundary
   serialises the resulting cross-thread race - all threads reading
   the shared byte observe the same race-winner, so the output has a
   well-defined grouping structure (e.g. out[2k] == out[2k+1] for
   i // 2). Unsafe fusion via same-thread RAW relaxation would replace
   the cross-thread shared read with a thread-local read of the
   thread's own write, breaking the grouping invariant (each thread
   sees its own value of i, so out[2k] = 2k != 2k+1 = out[2k+1]).

The tests below assert the appropriate invariant for each pattern.
Every non-injective test below would fail before the tightened
fingerprint landed (the same-thread RAW relaxation would have admitted
the fusion, breaking the grouping invariant).
"""

import numpy as np

import quadrants as qd

from tests import test_utils

N = 1024


def _run_combined(combined, arr_size):
    """Run the combined kernel and return the output array as numpy."""
    a = qd.ndarray(qd.f32, shape=arr_size)
    o = qd.ndarray(qd.f32, shape=N)
    combined(a, o)
    return o.to_numpy()


# ---------------------------------------------------------------------------
# Injective patterns: bijective addresses. Different threads touch
# different bytes, so the fused output is deterministic and matches
# what the per-thread arithmetic would produce.
# ---------------------------------------------------------------------------


@test_utils.test()
def test_fuse_raw_injective_identity():
    # arr[i] = i; out[i] = arr[i] = i
    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[i] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[i]

    out = _run_combined(wr, N)
    np.testing.assert_array_equal(out, np.arange(N, dtype=np.float32))


@test_utils.test()
def test_fuse_raw_injective_const_offset():
    # arr[i + 1] = i for i in [0, N-1); out[i] = arr[i + 1] = i
    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N - 1):
            a[i + 1] = qd.cast(i, qd.f32)
        for i in range(N - 1):
            o[i] = a[i + 1]

    out = _run_combined(wr, N)
    np.testing.assert_array_equal(out[: N - 1], np.arange(N - 1, dtype=np.float32))


@test_utils.test()
def test_fuse_raw_injective_const_mul():
    # arr[i * 2] = i; out[i] = arr[i * 2] = i
    M = N

    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N // 2):
            a[i * 2] = qd.cast(i, qd.f32)
        for i in range(N // 2):
            o[i] = a[i * 2]

    out = _run_combined(wr, M)
    np.testing.assert_array_equal(out[: N // 2], np.arange(N // 2, dtype=np.float32))


@test_utils.test()
def test_fuse_raw_injective_affine():
    # arr[2*i + 3] = i; out[i] = arr[2*i + 3] = i
    M = (2 * (N - 1) + 3) + 1

    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[2 * i + 3] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[2 * i + 3]

    out = _run_combined(wr, M)
    np.testing.assert_array_equal(out, np.arange(N, dtype=np.float32))


@test_utils.test()
def test_fuse_raw_injective_xor_const():
    # arr[i ^ 1] = i; out[i] = arr[i ^ 1] = i  (xor pairs up even/odd)
    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[i ^ 1] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[i ^ 1]

    out = _run_combined(wr, N)
    np.testing.assert_array_equal(out, np.arange(N, dtype=np.float32))


@test_utils.test()
def test_fuse_raw_injective_shl_const():
    # arr[i << 1] = i; out[i] = arr[i << 1] = i
    M = ((N - 1) << 1) + 1

    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[i << 1] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[i << 1]

    out = _run_combined(wr, M)
    np.testing.assert_array_equal(out, np.arange(N, dtype=np.float32))


# ---------------------------------------------------------------------------
# Non-injective patterns: multiple loop iterations map onto the same
# byte. The kernel-boundary fence between A and B means that, after A
# completes, every thread reads the same surviving race-winner from the
# shared cell. The fused output therefore has a well-defined grouping
# structure where every group of colliding threads sees the same value.
# Same-thread RAW relaxation would substitute each thread's own write
# for the shared read, breaking the grouping.
# ---------------------------------------------------------------------------


@test_utils.test()
def test_fuse_raw_noninjective_floordiv():
    # arr[i // 2]: threads 2k and 2k+1 both touch arr[k]. After the
    # write phase one of {2k, 2k+1} survives in arr[k]; both threads
    # then read the same survivor in the read phase, so out[2k] must
    # equal out[2k+1]. Unsafe fusion would give out[2k]=2k, out[2k+1]=2k+1.
    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[i // 2] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[i // 2]

    out = _run_combined(wr, N // 2)
    even = out[0:N:2]
    odd = out[1:N:2]
    np.testing.assert_array_equal(even, odd)
    # Each surviving value must be one of {2k, 2k+1} for its slot k.
    expected_lo = np.arange(0, N, 2, dtype=np.float32)
    expected_hi = np.arange(1, N, 2, dtype=np.float32)
    assert np.all((even == expected_lo) | (even == expected_hi)), (
        f"unexpected survivors: {even[:8].tolist()}"
    )


@test_utils.test()
def test_fuse_raw_noninjective_mod():
    # arr[i % 4]: 256 threads write each of arr[0..3]; all threads then
    # read the survivor in their slot. Group structure: out[i] == out[i + 4].
    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[i % 4] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[i % 4]

    out = _run_combined(wr, 4)
    for offset in range(4):
        group = out[offset::4]
        assert np.all(group == group[0]), (
            f"group {offset} not uniform: first 8 = {group[:8].tolist()}"
        )


@test_utils.test()
def test_fuse_raw_noninjective_bitand():
    # arr[i & 1]: two cells, all N threads write/read. out[i] == out[i + 2].
    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[i & 1] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[i & 1]

    out = _run_combined(wr, 2)
    even = out[0:N:2]
    odd = out[1:N:2]
    assert np.all(even == even[0]), (
        f"even group not uniform: first 8 = {even[:8].tolist()}"
    )
    assert np.all(odd == odd[0]), (
        f"odd group not uniform: first 8 = {odd[:8].tolist()}"
    )


@test_utils.test()
def test_fuse_raw_noninjective_min():
    # arr[min(i, K)]: threads i >= K all collapse to arr[K]. All
    # threads with i >= K must read the same value.
    K = 4

    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[qd.min(i, K)] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[qd.min(i, K)]

    out = _run_combined(wr, K + 1)
    clamped = out[K:]
    assert np.all(clamped == clamped[0]), (
        f"clamped tail not uniform: first 8 = {clamped[:8].tolist()}"
    )


@test_utils.test()
def test_fuse_raw_noninjective_constant_index():
    # arr[0]: no loop-index dependence at all. All threads write arr[0]
    # then all threads read arr[0]; every read must see the same
    # surviving race-winner. Unsafe fusion (substituting the thread's
    # own write for the shared read) would give out[i] = i, so the
    # output would be non-uniform.
    @qd.kernel
    def wr(a: qd.types.NDArray, o: qd.types.NDArray):
        for i in range(N):
            a[0] = qd.cast(i, qd.f32)
        for i in range(N):
            o[i] = a[0]

    out = _run_combined(wr, 1)
    assert np.all(out == out[0]), (
        f"out is non-uniform under arr[0]; first 8 = {out[:8].tolist()}"
    )
