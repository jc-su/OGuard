import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1024, 2053, 1000003)
TILE = 1024


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("tile_reverse_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    for start in range(0, values.shape[0], TILE):
        out[start : start + TILE] = values[start : start + TILE][::-1]
    return out


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_tile_reverse() -> None:
    mod = load()
    rng = np.random.default_rng(10)
    for n in SIZES:
        values = rng.integers(-(2**31), 2**31 - 1, size=n, dtype=np.int32)
        d_values = cuda.to_device(values)
        d_out = cuda.to_device(np.zeros(n, dtype=np.int32))
        mod.run(d_values, d_out, n)
        cuda.synchronize()
        got = d_out.copy_to_host()
        assert np.array_equal(got, reference(values)), n


if __name__ == "__main__":
    test_compile_ptx()
    test_tile_reverse()
    print("tile_reverse numba: PASS")
