"""GPU test of the Numba sorted_contains kernel against a NumPy reference.

Run from the model directory: python -P sorted_contains/test_numba.py
"""

import importlib.util
import pathlib
import warnings
from types import ModuleType

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Small test sizes launch few blocks; the low-occupancy warning is expected.
warnings.simplefilter("ignore", NumbaPerformanceWarning)

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)
MEMBER_SIZES = (0, 1, 2, 3, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sorted_contains_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_members(rng: np.random.Generator, s: int) -> np.ndarray:
    members = np.unique(rng.integers(-(2**40), 2**40, size=s, dtype=np.int64))
    while members.size < s:
        extra = rng.integers(-(2**40), 2**40, size=s - members.size, dtype=np.int64)
        members = np.unique(np.concatenate((members, extra)))
    return members


def make_keys(rng: np.random.Generator, n: int, members: np.ndarray) -> np.ndarray:
    keys = rng.integers(-(2**40), 2**40, size=n, dtype=np.int64)
    if members.size > 0:
        hit = rng.random(n) < 0.5
        keys[hit] = members[rng.integers(0, members.size, size=int(hit.sum()))]
        near = rng.random(n) < 0.2
        keys[near] = members[rng.integers(0, members.size, size=int(near.sum()))] + 1
    extremes = np.array([INT64_MIN, INT64_MAX], dtype=np.int64)
    keys[: min(n, 2)] = extremes[: min(n, 2)]
    return keys


def check(module: ModuleType, keys: np.ndarray, members: np.ndarray) -> None:
    n = keys.size
    s = members.size
    expected = np.isin(keys, members).astype(np.int8)
    d_keys = cuda.to_device(keys)
    d_members = cuda.to_device(members if s > 0 else np.zeros(1, dtype=np.int64))
    d_match = cuda.to_device(np.full(n, -1, dtype=np.int8))
    module.run(d_keys, d_members, d_match, n, s)
    np.testing.assert_array_equal(d_match.copy_to_host(), expected)


def test_sorted_contains() -> None:
    module = load_module()
    rng = np.random.default_rng(10)
    for s in MEMBER_SIZES:
        members = make_members(rng, s)
        for n in SIZES:
            check(module, make_keys(rng, n, members), members)
    # Members at the ends of the int64 range.
    members = np.array([INT64_MIN, -5, 0, 7, INT64_MAX], dtype=np.int64)
    keys = np.array([INT64_MIN, INT64_MIN + 1, -5, -4, 0, 6, 7, 8, INT64_MAX - 1, INT64_MAX])
    check(module, keys.astype(np.int64), members)


if __name__ == "__main__":
    test_sorted_contains()
    print("sorted_contains numba: passed")
