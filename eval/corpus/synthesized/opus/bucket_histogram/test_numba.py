import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
BOUND_COUNTS = (0, 1, 15, 4095, 10000)


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("bucket_histogram_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_bucket_histogram() -> None:
    mod = load()
    rng = np.random.default_rng(7)
    for n in SIZES:
        for b in BOUND_COUNTS:
            bounds = np.sort(rng.uniform(-10.0, 10.0, size=b))
            if b > 3:
                bounds[2] = bounds[1]
            values = rng.uniform(-12.0, 12.0, size=n)
            if b > 0 and n > 0:
                ties = rng.integers(0, n, size=n // 4 + 1)
                values[ties] = bounds[rng.integers(0, b, size=ties.shape[0])]
            buckets = np.searchsorted(bounds, values, side="right")
            expected = np.bincount(buckets, minlength=b + 1).astype(np.int32)
            d_values = cuda.to_device(values)
            d_bounds = cuda.to_device(bounds)
            d_counts = cuda.to_device(np.zeros(b + 1, dtype=np.int32))
            mod.run(d_values, d_bounds, d_counts, n, b)
            cuda.synchronize()
            got = d_counts.copy_to_host()
            assert np.array_equal(got, expected), (n, b)


if __name__ == "__main__":
    test_compile_ptx()
    test_bucket_histogram()
    print("bucket_histogram numba: PASS")
