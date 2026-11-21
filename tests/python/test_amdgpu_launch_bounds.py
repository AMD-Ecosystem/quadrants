"""Guards the AMDGPU launch bounds. Every offloaded task is launched with exactly `task.block_dim` threads per
workgroup, so codegen declares `amdgpu-flat-work-group-size = 1,<block_dim>` (the counterpart of CUDA's `maxntidx`)
together with `amdgpu-waves-per-eu = 2`. Without them the backend budgets registers for a 1024-thread workgroup, or,
with the block size alone, may use the whole register file for one wave per SIMD. Results stay correct either way, so
this asserts the IR attributes in addition to the results.
"""

import glob

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


def _ir(tmp_path):
    files = sorted(glob.glob(str(tmp_path / "quadrants_kernel_amdgpu_llvm_ir_*.ll")))
    assert files, "expected print_kernel_llvm_ir to emit a .ll dump, but none was found"
    return "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)


@pytest.mark.parametrize("block_dim", [32, 64, 256, 1024])
@test_utils.test(arch=qd.amdgpu, print_kernel_llvm_ir=True, offline_cache=False)
def test_amdgpu_declares_task_block_dim(block_dim, monkeypatch, tmp_path):
    # print_kernel_llvm_ir writes .ll dumps to the process CWD; isolate them.
    monkeypatch.chdir(tmp_path)

    n = 10000
    x = qd.ndarray(qd.f32, shape=n)

    @qd.kernel
    def fill(x: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        qd.loop_config(block_dim=block_dim)
        for i in range(n):
            x[i] = qd.sin(qd.cast(i, qd.f32)) * 2.0 + 1.0

    fill(x)
    np.testing.assert_allclose(x.to_numpy(), np.sin(np.arange(n, dtype=np.float32)) * 2.0 + 1.0, rtol=1e-5, atol=1e-5)

    ir = _ir(tmp_path)
    assert f'"amdgpu-flat-work-group-size"="1,{block_dim}"' in ir, (
        f"expected the task to declare amdgpu-flat-work-group-size=1,{block_dim}; IR follows:\n" + ir[-4000:]
    )
    assert '"amdgpu-waves-per-eu"="2"' in ir, (
        "expected the task to declare amdgpu-waves-per-eu=2; IR follows:\n" + ir[-4000:]
    )


@test_utils.test(arch=qd.amdgpu)
def test_amdgpu_launch_bounds_struct_for_and_reduction():
    # Struct-for and reduction tasks get codegen-chosen block sizes; their results must be unaffected.
    n = 4096
    a = qd.field(qd.f32)
    qd.root.dense(qd.i, n // 64).dense(qd.i, 64).place(a)
    total = qd.field(qd.f32, shape=())

    @qd.kernel
    def fill():
        for i in range(n):
            if i % 3 == 0:
                a[i] = 1.0

    @qd.kernel
    def reduce():
        for i in a:
            total[None] += a[i]

    fill()
    reduce()
    assert total[None] == len(range(0, n, 3))
