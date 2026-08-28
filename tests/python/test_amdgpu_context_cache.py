"""Regression coverage for the AMDGPU launcher's skip-redundant-RuntimeContext-HtoD optimization.

See ``quadrants/runtime/amdgpu/kernel_launcher.cpp``. On the default-stream path, repeated same-handle launches
reuse a cached ``RuntimeContext`` and skip the per-launch host->device copy. The optimization must be completely
transparent: identical, correct results regardless of whether a given launch skipped the upload. These tests
exercise the code paths that make the struct byte-stable so the cache can hit -- which is the subtle correctness
property, since the per-launch-varying data must still reach the device via the separately-uploaded ``arg_buffer``
rather than the cached ``RuntimeContext``:

  - argument-less, result-less kernels: ``arg_buffer`` is normalized to ``nullptr`` and the context is stable, so
    the cache hits after the first launch;
  - result-producing (reduction) kernels: ``result_buffer`` is pinned to the persistent device buffer, so the
    struct stays stable while the kernel still writes real results back;
  - kernels whose ndarray arguments change between launches: the changed device pointers live in ``arg_buffer``
    (uploaded every launch), while ``RuntimeContext`` holds only the *stable* device ``arg_buffer`` address and so
    stays byte-identical and keeps hitting the cache -- proving the cached-context / always-uploaded-arg_buffer
    split is sound.

The genuine forced-re-upload path -- a ``checkpoint_*_ptr`` mutation, which lives *inside* ``RuntimeContext`` and
therefore changes the compared bytes -- is covered by ``tests/python/test_checkpoint.py`` (streaming-launch
checkpoint state), so it is not duplicated here.

On non-AMDGPU backends the same kernels run through the generic launcher and act as a cross-backend correctness
baseline.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_repeated_launch_argless_result_less():
    # Argument-less, result-less kernel: exercises arg_buffer==nullptr normalization + cache-hit skip across
    # many identical launches. A stale/wrong skip would drop increments or read a dangling context.
    x = qd.field(qd.i32, shape=())
    x[None] = 0

    @qd.kernel
    def inc():
        x[None] += 1

    for _ in range(64):
        inc()
    assert x[None] == 64


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_repeated_launch_reduction_result_buffer():
    # Result-producing (reduction) kernel launched repeatedly: exercises the unconditional result_buffer pinning
    # plus the cache on a kernel that actually writes results back. The scalar arg changes each iteration, but it
    # rides the separately-uploaded arg_buffer -- RuntimeContext stays byte-stable and keeps hitting the cache --
    # so the returned reduction must still track the freshly filled data.
    n = 1024
    f = qd.field(qd.f32, shape=n)

    @qd.kernel
    def fill(c: qd.f32):
        for i in f:
            f[i] = c

    @qd.kernel
    def total() -> qd.f32:
        s = 0.0
        for i in f:
            s += f[i]
        return s

    for k in range(1, 17):
        fill(float(k))
        assert total() == pytest.approx(float(n * k))


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_repeated_launch_changing_ndarray_arg_buffer_split():
    # Same compiled handle launched against different ndarrays: the changed data pointers ride the separately-
    # uploaded arg_buffer, while RuntimeContext holds only the stable device arg_buffer address and keeps hitting
    # the cache. Verifies the cached-context / always-uploaded-arg_buffer split still observes the correct buffer.
    n = 256
    arrs = [qd.ndarray(qd.f32, shape=n) for _ in range(6)]
    for j, a in enumerate(arrs):
        a.from_numpy(np.full(n, float(j), dtype=np.float32))

    @qd.kernel
    def add_one(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(a.shape[0]):
            a[i] += 1.0

    for a in arrs:
        add_one(a)

    for j, a in enumerate(arrs):
        assert np.allclose(a.to_numpy(), float(j) + 1.0)
