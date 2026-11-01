import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("diff_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    if values.shape[0] > 0:
        out[0] = values[0]
        with np.errstate(over="ignore"):
            out[1:] = values[1:] - values[:-1]
    return out


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_diff() -> None:
    mod = load()
    rng = np.random.default_rng(8)
    for n in SIZES:
        values = rng.integers(-(2**40), 2**40, size=n, dtype=np.int64)
        d_values = cuda.to_device(values)
        d_out = cuda.to_device(np.full(n, 3, dtype=np.int64))
        mod.run(d_values, d_out, n)
        cuda.synchronize()
        got = d_out.copy_to_host()
        assert np.array_equal(got, reference(values)), n


if __name__ == "__main__":
    test_compile_ptx()
    test_diff()
    print("diff numba: PASS")
