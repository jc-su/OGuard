import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("run_starts_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values: np.ndarray) -> np.ndarray:
    flags = np.zeros(values.shape[0], dtype=np.int8)
    if values.shape[0] > 0:
        flags[0] = 1
        flags[1:] = values[1:] != values[:-1]
    return flags


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_run_starts() -> None:
    mod = load()
    rng = np.random.default_rng(1)
    for n in SIZES:
        values = np.sort(rng.integers(-50, 50, size=n, dtype=np.int64))
        values[: n // 2] = rng.integers(-3, 3, size=n // 2, dtype=np.int64)
        d_values = cuda.to_device(values)
        d_flags = cuda.to_device(np.full(n, 7, dtype=np.int8))
        mod.run(d_values, d_flags, n)
        cuda.synchronize()
        got = d_flags.copy_to_host()
        assert np.array_equal(got, reference(values)), n


if __name__ == "__main__":
    test_compile_ptx()
    test_run_starts()
    print("run_starts numba: PASS")
