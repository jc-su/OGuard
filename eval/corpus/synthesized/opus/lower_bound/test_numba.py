import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("lower_bound_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_lower_bound() -> None:
    mod = load()
    rng = np.random.default_rng(3)
    for n in SIZES:
        for s in (0, 1, 37, 100000):
            keys = rng.integers(-1000, 1000, size=n, dtype=np.int64)
            if n > 0:
                keys[0] = np.iinfo(np.int64).min
                keys[-1] = np.iinfo(np.int64).max
            sorted_ = np.sort(rng.integers(-900, 900, size=s, dtype=np.int64))
            expected = np.searchsorted(sorted_, keys, side="left").astype(np.int32)
            d_keys = cuda.to_device(keys)
            d_sorted = cuda.to_device(sorted_)
            d_out = cuda.to_device(np.full(n, -1, dtype=np.int32))
            mod.run(d_keys, d_sorted, d_out, n, s)
            cuda.synchronize()
            got = d_out.copy_to_host()
            assert np.array_equal(got, expected), (n, s)


if __name__ == "__main__":
    test_compile_ptx()
    test_lower_bound()
    print("lower_bound numba: PASS")
