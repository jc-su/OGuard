import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("segment_max_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_offsets(rng: np.random.Generator, g: int) -> np.ndarray:
    lengths = rng.integers(0, 9, size=g, dtype=np.int64)
    if g > 0:
        lengths[rng.integers(0, g)] = 5000
    offsets = np.zeros(g + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    return offsets


def reference(offsets: np.ndarray, values: np.ndarray) -> np.ndarray:
    g = offsets.shape[0] - 1
    maxes = np.full(g, INT64_MIN, dtype=np.int64)
    for j in range(g):
        if offsets[j + 1] > offsets[j]:
            maxes[j] = values[offsets[j] : offsets[j + 1]].max()
    return maxes


def reference_fast(offsets: np.ndarray, values: np.ndarray) -> np.ndarray:
    g = offsets.shape[0] - 1
    maxes = np.full(g, INT64_MIN, dtype=np.int64)
    nonempty = offsets[1:] > offsets[:-1]
    if values.shape[0] > 0:
        starts = offsets[:-1][nonempty]
        maxes[nonempty] = np.maximum.reduceat(values, starts)
    return maxes


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_segment_max() -> None:
    mod = load()
    rng = np.random.default_rng(2)
    for g in SIZES:
        offsets = make_offsets(rng, g)
        n = int(offsets[g])
        values = rng.integers(INT64_MIN, np.iinfo(np.int64).max, size=n, dtype=np.int64)
        expected = reference(offsets, values) if g <= 1000 else reference_fast(offsets, values)
        d_offsets = cuda.to_device(offsets)
        d_values = cuda.to_device(values)
        d_maxes = cuda.to_device(np.full(g, 5, dtype=np.int64))
        mod.run(d_offsets, d_values, d_maxes, g)
        cuda.synchronize()
        got = d_maxes.copy_to_host()
        assert np.array_equal(got, expected), g


if __name__ == "__main__":
    test_compile_ptx()
    test_segment_max()
    print("segment_max numba: PASS")
