import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("scatter_add_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_scatter_add() -> None:
    mod = load()
    rng = np.random.default_rng(9)
    for m in SIZES:
        for n in SIZES:
            if m > 0 and n == 0:
                continue
            idx = rng.integers(0, max(n, 1), size=m, dtype=np.int32)
            # Multiples of 1/4 with small magnitude add exactly in any order.
            values = rng.integers(-4000, 4000, size=m).astype(np.float64) / 4.0
            expected = np.zeros(n, dtype=np.float64)
            np.add.at(expected, idx, values)
            d_idx = cuda.to_device(idx)
            d_values = cuda.to_device(values)
            d_out = cuda.to_device(np.zeros(n, dtype=np.float64))
            mod.run(d_idx, d_values, d_out, m, n)
            cuda.synchronize()
            got = d_out.copy_to_host()
            assert np.array_equal(got, expected), (m, n)


if __name__ == "__main__":
    test_compile_ptx()
    test_scatter_add()
    print("scatter_add numba: PASS")
