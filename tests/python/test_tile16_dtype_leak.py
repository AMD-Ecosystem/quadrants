"""
Regression test: Tile16x16 dtype must track the active default_fp, not the
dtype from when the module was first imported.

make_tile16x16() caches classes by dtype. If a caller captures the result at
module level (e.g. `Tile16x16 = make_tile16x16(qd.f64)`) and then switches
default_fp to f32 via qd.init()/qd.reset(), the stale f64 class silently
computes in double precision and truncates on store — a ~2.4x slowdown on
GPUs with slow f64.
"""

import numpy as np

import quadrants as qd
from quadrants.lang.simt.tile16 import make_tile16x16

from tests import test_utils

N = 16


@test_utils.test(arch=[qd.cuda])
def test_tile16_f64_roundtrip_into_f32_array():
    """Load f32 data through an f64 tile and store back — must be lossless."""

    Tile_f64 = make_tile16x16(qd.f64)
    Tile_f32 = make_tile16x16(qd.f32)

    src = qd.ndarray(shape=(N, N), dtype=qd.f32)
    dst_f32 = qd.ndarray(shape=(N, N), dtype=qd.f32)
    dst_f64 = qd.ndarray(shape=(N, N), dtype=qd.f32)

    Ann = qd.types.NDArray[qd.f32, 2]

    @qd.kernel
    def roundtrip_f32(s: Ann, d: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile_f32()
            t[:] = s[0:N, 0:N]
            d[0:N, 0:N] = t

    @qd.kernel
    def roundtrip_f64(s: Ann, d: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile_f64()
            t[:] = s[0:N, 0:N]
            d[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)

    roundtrip_f32(src, dst_f32)
    roundtrip_f64(src, dst_f64)

    np.testing.assert_array_equal(dst_f32.to_numpy(), data)
    np.testing.assert_array_equal(dst_f64.to_numpy(), data)
