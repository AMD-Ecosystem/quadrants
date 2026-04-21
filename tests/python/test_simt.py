import numpy as np
import pytest
from pytest import approx

import quadrants as qd
from quadrants.lang.simt import subgroup

from tests import test_utils


@test_utils.test(arch=qd.cuda)
def test_all_nonzero():
    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.all_nonzero(qd.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 1
        a[i] = -1

    foo()

    for i in range(32):
        assert a[i] == 1

    b[np.random.randint(0, 32)] = 0

    foo()

    for i in range(32):
        assert a[i] == 0


@test_utils.test(arch=qd.cuda)
def test_sync_all_nonzero():
    a = qd.field(dtype=qd.i32, shape=256)
    b = qd.field(dtype=qd.i32, shape=256)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=256)
        for i in range(256):
            a[i] = qd.simt.block.sync_all_nonzero(b[i])

    for i in range(256):
        b[i] = 1
        a[i] = -1

    foo()

    for i in range(256):
        assert a[i] == 1

    b[np.random.randint(0, 256)] = 0

    foo()

    for i in range(256):
        assert a[i] == 0


@test_utils.test(arch=qd.cuda)
def test_any_nonzero():
    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.any_nonzero(qd.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 0
        a[i] = -1

    foo()

    for i in range(32):
        assert a[i] == 0

    b[np.random.randint(0, 32)] = 1

    foo()

    for i in range(32):
        assert a[i] == 1


@test_utils.test(arch=qd.cuda)
def test_sync_any_nonzero():
    a = qd.field(dtype=qd.i32, shape=256)
    b = qd.field(dtype=qd.i32, shape=256)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=256)
        for i in range(256):
            a[i] = qd.simt.block.sync_any_nonzero(b[i])

    for i in range(256):
        b[i] = 0
        a[i] = -1

    foo()

    for i in range(256):
        assert a[i] == 0

    b[np.random.randint(0, 256)] = 1

    foo()

    for i in range(256):
        assert a[i] == 1


@test_utils.test(arch=qd.cuda)
def test_sync_count_nonzero():
    a = qd.field(dtype=qd.i32, shape=256)
    b = qd.field(dtype=qd.i32, shape=256)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=256)
        for i in range(256):
            a[i] = qd.simt.block.sync_count_nonzero(b[i])

    for i in range(256):
        b[i] = 0
        a[i] = -1

    foo()

    for i in range(256):
        assert a[i] == 0

    random_idx_count = np.random.randint(0, 256)
    random_idx = np.random.choice(256, random_idx_count, replace=False)
    for i in range(random_idx_count):
        b[random_idx[i]] = 1

    foo()

    for i in range(256):
        assert a[i] == random_idx_count


@test_utils.test(arch=qd.cuda)
def test_unique():
    a = qd.field(dtype=qd.u32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def check():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.unique(qd.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 0
        a[i] = -1

    check()

    for i in range(32):
        assert a[i] == 1

    for i in range(32):
        b[i] = i + 100

    check()

    for i in range(32):
        assert a[i] == 1

    b[np.random.randint(0, 32)] = 0

    check()

    for i in range(32):
        assert a[i] == 0


@test_utils.test(arch=qd.cuda)
def test_ballot():
    a = qd.field(dtype=qd.u32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.ballot(b[i])

    key = 0
    for i in range(32):
        b[i] = i % 2
        key += b[i] * pow(2, i)

    foo()

    for i in range(32):
        assert a[i] == key


@test_utils.test(arch=qd.cuda)
def test_shfl_sync_i32():
    a = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_sync_i32(qd.u32(0xFFFFFFFF), a[i], 0)

    for i in range(32):
        a[i] = i + 1

    foo()

    for i in range(1, 32):
        assert a[i] == 1


@test_utils.test(arch=qd.cuda)
def test_shfl_sync_f32():
    a = qd.field(dtype=qd.f32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_sync_f32(qd.u32(0xFFFFFFFF), a[i], 0)

    for i in range(32):
        a[i] = i + 1.0

    foo()

    for i in range(1, 32):
        assert a[i] == approx(1.0, abs=1e-4)


@test_utils.test(arch=qd.cuda)
def test_shfl_up_i32():
    # TODO
    pass


@test_utils.test(arch=qd.cuda)
def test_shfl_xor_i32():
    a = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            for j in range(5):
                offset = 1 << j
                a[i] += qd.simt.warp.shfl_xor_i32(qd.u32(0xFFFFFFFF), a[i], offset)

    value = 0
    for i in range(32):
        a[i] = i
        value += i

    foo()

    for i in range(32):
        assert a[i] == value


@test_utils.test(arch=qd.cuda)
def test_shfl_down_i32():
    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_down_i32(qd.u32(0xFFFFFFFF), b[i], 1)

    for i in range(32):
        b[i] = i * i

    foo()

    for i in range(31):
        assert a[i] == b[i + 1]

    # TODO: make this test case stronger


@test_utils.test(arch=qd.cuda)
def test_shfl_up_i32():
    a = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_up_i32(qd.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i

    foo()

    for i in range(1, 32):
        assert a[i] == (i - 1) * (i - 1)


@test_utils.test(arch=qd.cuda)
def test_shfl_up_f32():
    a = qd.field(dtype=qd.f32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_up_f32(qd.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i * 0.9

    foo()

    for i in range(1, 32):
        assert a[i] == approx((i - 1) * (i - 1) * 0.9, abs=1e-4)


@test_utils.test(arch=qd.cuda)
def test_shfl_down_f32():
    a = qd.field(dtype=qd.f32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_down_f32(qd.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i * 0.9

    foo()

    for i in range(31):
        assert a[i] == approx((i + 1) * (i + 1) * 0.9, abs=1e-4)


@test_utils.test(arch=qd.cuda)
def test_match_any():
    # Skip match_any test for Pascal
    if qd.lang.impl.get_cuda_compute_capability() < 70:
        pytest.skip("match_any not supported on Pascal")

    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.u32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(16):
            a[i] = 0
            a[i + 16] = 1

        for i in range(32):
            b[i] = qd.simt.warp.match_any(qd.u32(0xFFFFFFFF), a[i])

    foo()

    for i in range(16):
        assert b[i] == 65535
    for i in range(16):
        assert b[i + 16] == (2**32 - 2**16)


@test_utils.test(arch=qd.cuda)
def test_match_all():
    # Skip match_all test for Pascal
    if qd.lang.impl.get_cuda_compute_capability() < 70:
        pytest.skip("match_all not supported on Pascal")

    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.u32, shape=32)
    c = qd.field(dtype=qd.u32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = 1
        for i in range(32):
            b[i] = qd.simt.warp.match_all(qd.u32(0xFFFFFFFF), a[i])

        a[0] = 2
        for i in range(32):
            c[i] = qd.simt.warp.match_all(qd.u32(0xFFFFFFFF), a[i])

    foo()

    for i in range(32):
        assert b[i] == (2**32 - 1)

    for i in range(32):
        assert c[i] == 0


@test_utils.test(arch=qd.cuda)
def test_active_mask():
    a = qd.field(dtype=qd.u32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=16)
        for i in range(32):
            a[i] = qd.simt.warp.active_mask()

    foo()

    for i in range(32):
        assert a[i] == 65535


@test_utils.test(arch=qd.cuda)
def test_warp_sync():
    a = qd.field(dtype=qd.u32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = i
        qd.simt.warp.sync(qd.u32(0xFFFFFFFF))
        for i in range(16):
            a[i] = a[i + 16]

    foo()

    for i in range(32):
        assert a[i] == i % 16 + 16


@test_utils.test(arch=qd.cuda)
def test_block_sync():
    N = 1024
    a = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            # Make the 0-th thread runs slower intentionally
            for j in range(N - i):
                a[i] = j
            qd.simt.block.sync()
            if i > 0:
                a[i] = a[0]

    foo()

    for i in range(N):
        assert a[i] == N - 1


# TODO: replace this with a stronger test case
@test_utils.test(arch=qd.cuda)
def test_grid_memfence():
    N = 1000
    BLOCK_SIZE = 1
    a = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        block_counter = 0
        qd.loop_config(block_dim=BLOCK_SIZE)
        for i in range(N):
            a[i] = 1
            qd.simt.grid.memfence()

            # Execute a prefix sum after all blocks finish
            actual_order_of_block = qd.atomic_add(block_counter, 1)
            if actual_order_of_block == N - 1:
                for j in range(1, N):
                    a[j] += a[j - 1]

    foo()

    for i in range(N):
        assert a[i] == i + 1


# Higher level primitives test
def _test_subgroup_reduce(op, group_op, np_op, size, initial_value, dtype):
    field = qd.field(dtype, (size))
    if dtype == qd.i32 or dtype == qd.i64:
        rand_values = np.random.randint(1, 100, size=(size))
        field.from_numpy(rand_values)
    if dtype == qd.f32 or dtype == qd.f64:
        rand_values = np.random.random(size=(size)).astype(np.float32)
        field.from_numpy(rand_values)

    @qd.kernel
    def reduce_all() -> dtype:
        sum = qd.cast(initial_value, dtype)
        for i in field:
            value = field[i]
            reduce_value = group_op(value)
            if subgroup.elect():
                op(sum, reduce_value)
        return sum

    if dtype == qd.i32 or dtype == qd.i64:
        assert reduce_all() == np_op(rand_values)
    else:
        assert reduce_all() == approx(np_op(rand_values), 3e-4)


# We use 2677 as size because it is a prime number
# i.e. any device other than a subgroup size of 1 should have one non active group


@test_utils.test(arch=qd.vulkan, exclude=[(qd.vulkan, "Darwin")])
def test_subgroup_reduction_add_i32():
    _test_subgroup_reduce(qd.atomic_add, subgroup.reduce_add, np.sum, 2677, 0, qd.i32)


@test_utils.test(arch=qd.vulkan)
def test_subgroup_reduction_add_f32():
    _test_subgroup_reduce(qd.atomic_add, subgroup.reduce_add, np.sum, 2677, 0, qd.f32)


# @test_utils.test(arch=qd.vulkan)
# def test_subgroup_reduction_mul_i32():
#     _test_subgroup_reduce(qd.atomic_add, subgroup.reduce_mul, np.prod, 8, 1, qd.f32)


@test_utils.test(arch=qd.vulkan, exclude=[(qd.vulkan, "Darwin")])
def test_subgroup_reduction_max_i32():
    _test_subgroup_reduce(qd.atomic_max, subgroup.reduce_max, np.max, 2677, 0, qd.i32)


@test_utils.test(arch=qd.vulkan)
def test_subgroup_reduction_max_f32():
    _test_subgroup_reduce(qd.atomic_max, subgroup.reduce_max, np.max, 2677, 0, qd.f32)


@test_utils.test(arch=qd.vulkan)
def test_subgroup_reduction_min_f32():
    _test_subgroup_reduce(qd.atomic_max, subgroup.reduce_max, np.max, 2677, 0, qd.f32)


# =============================================================================
# AMDGPU wave64 shuffle tests
# =============================================================================
# These tests validate warp shuffle operations on AMDGPU with 64-lane wavefronts.
# The butterfly reduction with offsets [32, 16, 8, 4, 2, 1] reduces all 64 lanes.
# For shfl_xor: every lane ends up with the full sum.
# For shfl_down: only lane 0 ends up with the full sum.


@test_utils.test(arch=qd.amdgpu)
def test_shfl_xor_f32_wave64_butterfly():
    """Butterfly sum via shfl_xor_f32 on wave64. Every lane should hold sum(0..63) = 2016."""
    WARP = 64
    result = qd.field(dtype=qd.f32, shape=WARP)

    @qd.kernel
    def butterfly_xor_sum():
        qd.loop_config(block_dim=WARP)
        for i in range(WARP):
            val = qd.cast(i, qd.f32)
            # 6-round butterfly with xor - all lanes get the sum
            for offset in qd.static([32, 16, 8, 4, 2, 1]):
                val = val + qd.simt.warp.shfl_xor_f32(qd.u32(0xFFFFFFFF), val, offset)
            result[i] = val

    butterfly_xor_sum()

    # sum(0..63) = 63*64/2 = 2016
    for i in range(WARP):
        assert result[i] == approx(2016.0, abs=1e-3), f"Lane {i}: expected 2016, got {result[i]}"


@test_utils.test(arch=qd.amdgpu)
def test_shfl_xor_f32_wave64_butterfly_fractional():
    """Butterfly sum via shfl_xor_f32 with fractional values to verify bitcast round-trip.

    Uses values with non-trivial fractional parts (lane * 0.5 + 0.125) to ensure
    the f32 -> i32 bitcast -> ds.bpermute -> i32 -> f32 bitcast preserves IEEE 754
    bit patterns correctly through all 6 shuffle rounds.
    """
    WARP = 64
    result = qd.field(dtype=qd.f32, shape=WARP)

    @qd.kernel
    def butterfly_xor_sum_fractional():
        qd.loop_config(block_dim=WARP)
        for i in range(WARP):
            # Fractional values: 0.125, 0.625, 1.125, ..., 31.625
            val = qd.cast(i, qd.f32) * 0.5 + 0.125
            # 6-round butterfly with xor - all lanes get the sum
            for offset in qd.static([32, 16, 8, 4, 2, 1]):
                val = val + qd.simt.warp.shfl_xor_f32(qd.u32(0xFFFFFFFF), val, offset)
            result[i] = val

    butterfly_xor_sum_fractional()

    # sum(i * 0.5 + 0.125 for i in 0..63) = 0.5 * 2016 + 64 * 0.125 = 1008 + 8 = 1016
    for i in range(WARP):
        assert result[i] == approx(1016.0, abs=1e-2), f"Lane {i}: expected 1016.0, got {result[i]}"


@test_utils.test(arch=qd.amdgpu)
def test_shfl_xor_i32_wave64_butterfly():
    """Butterfly sum via shfl_xor_i32 on wave64. Every lane should hold sum(0..63) = 2016."""
    WARP = 64
    result = qd.field(dtype=qd.i32, shape=WARP)

    @qd.kernel
    def butterfly_xor_sum():
        qd.loop_config(block_dim=WARP)
        for i in range(WARP):
            val = i
            # 6-round butterfly with xor - all lanes get the sum
            for offset in qd.static([32, 16, 8, 4, 2, 1]):
                val = val + qd.simt.warp.shfl_xor_i32(qd.u32(0xFFFFFFFF), val, offset)
            result[i] = val

    butterfly_xor_sum()

    # sum(0..63) = 63*64/2 = 2016
    for i in range(WARP):
        assert result[i] == 2016, f"Lane {i}: expected 2016, got {result[i]}"


@test_utils.test(arch=qd.amdgpu)
def test_shfl_down_f32_wave64_butterfly():
    """Butterfly sum via shfl_down_f32 on wave64. Lane 0 should hold sum(0..63) = 2016."""
    WARP = 64
    result = qd.field(dtype=qd.f32, shape=WARP)

    @qd.kernel
    def butterfly_down_sum():
        qd.loop_config(block_dim=WARP)
        for i in range(WARP):
            val = qd.cast(i, qd.f32)
            # 6-round butterfly with down - only lane 0 gets the full sum
            for offset in qd.static([32, 16, 8, 4, 2, 1]):
                val = val + qd.simt.warp.shfl_down_f32(qd.u32(0xFFFFFFFF), val, offset)
            result[i] = val

    butterfly_down_sum()

    # Lane 0 should have the full sum
    assert result[0] == approx(2016.0, abs=1e-3), f"Lane 0: expected 2016, got {result[0]}"


@test_utils.test(arch=qd.amdgpu)
def test_shfl_down_i32_wave64_butterfly():
    """Butterfly sum via shfl_down_i32 on wave64. Lane 0 should hold sum(0..63) = 2016."""
    WARP = 64
    result = qd.field(dtype=qd.i32, shape=WARP)

    @qd.kernel
    def butterfly_down_sum():
        qd.loop_config(block_dim=WARP)
        for i in range(WARP):
            val = i
            # 6-round butterfly with down - only lane 0 gets the full sum
            for offset in qd.static([32, 16, 8, 4, 2, 1]):
                val = val + qd.simt.warp.shfl_down_i32(qd.u32(0xFFFFFFFF), val, offset)
            result[i] = val

    butterfly_down_sum()

    # Lane 0 should have the full sum
    assert result[0] == 2016, f"Lane 0: expected 2016, got {result[0]}"


@test_utils.test(arch=qd.amdgpu)
def test_shfl_sync_f32_wave64():
    """Test shfl_sync_f32 (broadcast from specific lane) on wave64."""
    WARP = 64
    result = qd.field(dtype=qd.f32, shape=WARP)

    @qd.kernel
    def broadcast_from_lane_42():
        qd.loop_config(block_dim=WARP)
        for i in range(WARP):
            val = qd.cast(i * 2, qd.f32)  # Each lane has its index * 2
            # Broadcast from lane 42 (which has value 84.0)
            val = qd.simt.warp.shfl_sync_f32(qd.u32(0xFFFFFFFF), val, 42)
            result[i] = val

    broadcast_from_lane_42()

    # All lanes should have value from lane 42 = 84.0
    for i in range(WARP):
        assert result[i] == approx(84.0, abs=1e-4), f"Lane {i}: expected 84.0, got {result[i]}"


@test_utils.test(arch=qd.amdgpu)
def test_shfl_sync_i32_wave64():
    """Test shfl_sync_i32 (broadcast from specific lane) on wave64."""
    WARP = 64
    result = qd.field(dtype=qd.i32, shape=WARP)

    @qd.kernel
    def broadcast_from_lane_42():
        qd.loop_config(block_dim=WARP)
        for i in range(WARP):
            val = i * 2  # Each lane has its index * 2
            # Broadcast from lane 42 (which has value 84)
            val = qd.simt.warp.shfl_sync_i32(qd.u32(0xFFFFFFFF), val, 42)
            result[i] = val

    broadcast_from_lane_42()

    # All lanes should have value from lane 42 = 84
    for i in range(WARP):
        assert result[i] == 84, f"Lane {i}: expected 84, got {result[i]}"


@test_utils.test(arch=qd.amdgpu)
def test_shfl_xor_f32_wave64_asymmetric():
    """Test shfl_xor_f32 with asymmetric values to catch wrap-around bugs."""
    WARP = 64
    result = qd.field(dtype=qd.f32, shape=WARP)
    input_vals = qd.field(dtype=qd.f32, shape=WARP)

    # Use prime-spaced values that would fail if wrap-around is wrong
    for i in range(WARP):
        input_vals[i] = (i * 17 + 3) % 100 + 0.5

    @qd.kernel
    def asymmetric_xor():
        qd.loop_config(block_dim=WARP)
        for i in range(WARP):
            val = input_vals[i]
            # Single xor with offset 32 - swaps upper and lower halves
            val = qd.simt.warp.shfl_xor_f32(qd.u32(0xFFFFFFFF), val, 32)
            result[i] = val

    asymmetric_xor()

    # Lane i should have value from lane (i ^ 32)
    for i in range(WARP):
        expected = input_vals[i ^ 32]
        assert result[i] == approx(expected, abs=1e-4), \
            f"Lane {i}: expected {expected} from lane {i ^ 32}, got {result[i]}"
