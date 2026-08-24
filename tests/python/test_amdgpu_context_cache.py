"""Regression coverage for the AMDGPU launcher's skip-redundant-RuntimeContext-HtoD optimization.

See ``quadrants/runtime/amdgpu/kernel_launcher.cpp``. On the default-stream path, repeated same-handle launches
reuse a cached ``RuntimeContext`` and skip the per-launch host->device copy. The optimization must be completely
transparent: identical, correct results regardless of whether a given launch skipped the upload. These tests
exercise the code paths that make the struct byte-stable (so the cache can hit) and the path that must force a
re-upload:

  - argument-less, result-less kernels: ``arg_buffer`` is normalized to ``nullptr`` and the context is stable, so
    the cache hits after the first launch;
  - result-producing (reduction) kernels: ``result_buffer`` is pinned to the persistent device buffer, so the
    struct stays stable while the kernel still writes real results back;
  - kernels whose ndarray arguments change between launches: the changed device pointers live in the compared
    bytes, so each launch must re-upload and observe the correct buffer.

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
    # plus the cache on a kernel that actually writes results back. The scalar arg changes each iteration, so the
    # arg bytes differ and a re-upload is forced; the returned reduction must track the freshly filled data.
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
def test_repeated_launch_changing_ndarray_forces_reupload():
    # Same compiled handle launched against different ndarrays: the changed data pointer sits in the compared
    # RuntimeContext bytes, so each launch must re-upload rather than reuse the previous buffer's pointer.
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
