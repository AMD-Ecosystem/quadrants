"""Tests for the cross-platform occupancy hint ``@qd.kernel(min_blocks_per_cu=N)``.

Covers the parts of the feature that are hardware-independent and run in normal
CI (decoration-time validation and cache-key participation), plus a GPU-gated
smoke test that the hint compiles and runs on backends with an occupancy
concept (CUDA / AMDGPU). The exact lowering (``minctasm`` on CUDA,
``amdgpu-waves-per-eu`` on AMDGPU) is verified separately at the IR/PTX level;
here we only assert the plumbing is wired and does not alter results.
"""
import pytest

import quadrants as qd
from quadrants.lang._fast_caching import src_hasher
from quadrants.lang._wrap_inspect import get_source_info_and_src
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


# ------------------------------------------------------------------ validation


@test_utils.test(arch=qd.cpu)
def test_min_blocks_per_cu_accepts_valid():
    # None (default) and positive ints must decorate without error.
    @qd.kernel(min_blocks_per_cu=None)
    def k_none(x: qd.types.ndarray()):
        x[None] = x[None] + 1

    @qd.kernel(min_blocks_per_cu=1)
    def k_one(x: qd.types.ndarray()):
        x[None] = x[None] + 1

    @qd.kernel(min_blocks_per_cu=4)
    def k_four(x: qd.types.ndarray()):
        x[None] = x[None] + 1

    assert k_none._primal.min_blocks_per_cu is None
    assert k_one._primal.min_blocks_per_cu == 1
    assert k_four._primal.min_blocks_per_cu == 4


@test_utils.test(arch=qd.cpu)
@pytest.mark.parametrize("bad", [0, -1, -8, True, False, 1.5, "4"])
def test_min_blocks_per_cu_rejects_invalid(bad):
    # 0 / negatives / bool / non-int must raise at decoration time.
    with pytest.raises(qd.QuadrantsSyntaxError, match="min_blocks_per_cu"):

        @qd.kernel(min_blocks_per_cu=bad)
        def k(x: qd.types.ndarray()):
            x[None] = x[None] + 1


# ------------------------------------------------------------------ cache key


def _plain_kernel_fn(x: qd.types.ndarray()):
    x[None] = x[None] + 1


@test_utils.test(arch=qd.cpu)
def test_min_blocks_per_cu_participates_in_cache_key():
    # Using one shared function object means the ONLY differing input to
    # create_cache_key is min_blocks_per_cu, so distinct keys prove the hint
    # participates (and equal values are stable) -- guarding against a kernel
    # compiled with one value silently reusing another value's cached binary.
    info, _src = get_source_info_and_src(_plain_kernel_fn)

    def key(mbpc):
        return src_hasher.create_cache_key(
            raise_on_templated_floats=False,
            kernel_source_info=info,
            args=(),
            arg_metas=[],
            min_blocks_per_cu=mbpc,
        )

    k_none, k2, k4 = key(None), key(2), key(4)

    # keys were actually produced (fastcache not skipped for this no-arg kernel)
    assert k_none and k2 and k4
    # distinct values -> distinct keys
    assert len({k_none, k2, k4}) == 3
    # same value -> stable key
    assert key(4) == k4
    assert key(None) == k_none


# ------------------------------------------------------------------ GPU smoke


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_min_blocks_per_cu_runs_on_gpu():
    # End-to-end: the hint must not change numerical results on backends that
    # honor it (CUDA -> minctasm, AMDGPU -> amdgpu-waves-per-eu).
    @qd.kernel(min_blocks_per_cu=4)
    def add_one(x: qd.types.ndarray()):
        for i in x:
            x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(256,))
    x.fill(0)
    add_one(x)
    assert (x.to_numpy() == 1).all()
