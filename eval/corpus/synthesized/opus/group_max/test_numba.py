import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("group_max_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_group_max() -> None:
    mod = load()
    rng = np.random.default_rng(4)
    for n in SIZES:
        for capacity in (1, 17, 200000):
            group = rng.integers(0, capacity, size=n, dtype=np.int32)
            value = rng.integers(INT64_MIN, INT64_MAX, size=n, dtype=np.int64)
            best = np.full(capacity, INT64_MIN, dtype=np.int64)
            expected = best.copy()
            np.maximum.at(expected, group, value)
            d_group = cuda.to_device(group)
            d_value = cuda.to_device(value)
            d_best = cuda.to_device(best)
            mod.run(d_group, d_value, d_best, n, capacity)
            cuda.synchronize()
            got = d_best.copy_to_host()
            assert np.array_equal(got, expected), (n, capacity)


if __name__ == "__main__":
    test_compile_ptx()
    test_group_max()
    print("group_max numba: PASS")
