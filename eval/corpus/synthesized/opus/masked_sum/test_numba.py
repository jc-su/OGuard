import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("masked_sum_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_masked_sum() -> None:
    mod = load()
    rng = np.random.default_rng(6)
    for n in SIZES:
        values = rng.uniform(-100.0, 100.0, size=n)
        mask = rng.integers(0, 2, size=n, dtype=np.int8) * rng.integers(
            1, 127, size=n, dtype=np.int8
        )
        selected = values[mask != 0]
        expected = float(np.sum(selected))
        bound = 1e-12 * (float(np.sum(np.abs(selected))) + 1.0)
        d_values = cuda.to_device(values)
        d_mask = cuda.to_device(mask)
        d_result = cuda.to_device(np.zeros(1, dtype=np.float64))
        mod.run(d_values, d_mask, d_result, n)
        cuda.synchronize()
        got = float(d_result.copy_to_host()[0])
        assert abs(got - expected) <= bound, (n, got, expected)


if __name__ == "__main__":
    test_compile_ptx()
    test_masked_sum()
    print("masked_sum numba: PASS")
