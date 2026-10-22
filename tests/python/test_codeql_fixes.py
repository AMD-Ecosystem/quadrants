"""Exercise the CodeQL fixes that the AMDGPU runtime can reach from Python.

The float16 host write is the ndarray.cpp change. Compiling the constant matrix
index runs the int64 index multiply in frontend_ir.cpp and the AMDGPU ld.lld
launch in jit_amdgpu.cpp. copy_matrix_across_offload stores a tensor value in
the serial task and reads it from a parallel for, which is the size_t multiply
in transforms/offload.cpp.
"""

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.amdgpu)
def test_f16_ndarray_host_write():
    values = qd.ndarray(dtype=qd.f16, shape=(2,))
    values[0] = 1.0
    values[1] = -2.0
    assert abs(float(values[0]) - 1.0) < 1e-3
    assert abs(float(values[1]) + 2.0) < 1e-3


@test_utils.test(arch=qd.amdgpu)
def test_constant_tensor_index_and_shared_matrix():
    result = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def run():
        matrix = qd.Matrix([[1.0, 2.0], [3.0, 4.0]])
        result[None] = matrix[1, 0]
        for _ in range(1):
            result[None] = result[None] + matrix[0, 1]
        for _ in range(1):
            result[None] = result[None] + matrix[1, 1]

    run()
    assert result[None] == 9.0


@test_utils.test(arch=qd.amdgpu)
def test_tensor_value_crosses_offload():
    count = 4
    dst = qd.Matrix.field(2, 2, dtype=qd.f32, shape=count)

    @qd.kernel
    def copy_matrix_across_offload():
        mat = qd.Matrix([[1.0, 2.0], [3.0, 4.0]])
        for i in range(count):
            dst[i] = mat

    copy_matrix_across_offload()
    for i in range(count):
        got = dst[i]
        assert float(got[0, 0]) == 1.0
        assert float(got[0, 1]) == 2.0
        assert float(got[1, 0]) == 3.0
        assert float(got[1, 1]) == 4.0
