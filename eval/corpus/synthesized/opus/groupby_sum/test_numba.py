"""GPU test of the Numba groupby_sum kernel against a NumPy reference.

Run from the parent directory: `python -P groupby_sum/test_numba.py`. The -P
flag keeps this directory off sys.path, so its numba.py does not shadow Numba.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

SIZES = (0, 1, 1000, 1000003)
U32 = (1 << 32) - 1


def load_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("numba.py")
    spec = importlib.util.spec_from_file_location("groupby_sum_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table_capacity(groups: int) -> int:
    capacity = 1
    while capacity < 2 * groups:
        capacity *= 2
    return capacity


def distinct_groups(count: int, rng: np.random.Generator) -> npt.NDArray[np.int32]:
    """Distinct groups, negative ones included, none equal to -1.

    An odd multiplier is a bijection on 31-bit integers.
    """
    base = rng.permutation(count).astype(np.int64) + 777
    groups = ((base * 0x2545F491) & 0x7FFFFFFF) - (1 << 30)
    groups[groups == -1] = 1 << 30
    return groups.astype(np.int32)


def fmix32(group: int) -> int:
    h = group & U32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & U32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & U32
    h ^= h >> 16
    return h


def fmix32_inverse(h: int) -> int:
    x = h ^ (h >> 16)
    x = (x * pow(0xC2B2AE35, -1, 1 << 32)) & U32
    x ^= (x >> 13) ^ (x >> 26)
    x = (x * pow(0x85EBCA6B, -1, 1 << 32)) & U32
    x ^= x >> 16
    return x - (1 << 32) if x >= 1 << 31 else x


def colliding_groups(count: int, capacity: int) -> npt.NDArray[np.int32]:
    """Distinct groups that all have the home slot capacity - 2, so probes wrap."""
    groups = [fmix32_inverse((i + 1) * capacity + capacity - 2) for i in range(count)]
    assert -1 not in groups
    assert all(fmix32(g) & (capacity - 1) == capacity - 2 for g in groups)
    return np.array(groups, dtype=np.int32)


def aggregate_and_check(
    module: ModuleType,
    group: npt.NDArray[np.int32],
    value: npt.NDArray[np.float64],
    capacity: int,
    launch: tuple[CUDADispatcher, int, int] | None,
) -> None:
    """Aggregate with module.run, or with the given kernel, grid, and block size."""
    n = group.size
    d_table_keys = cuda.to_device(np.full(capacity, -1, dtype=np.int32))
    d_table_sums = cuda.to_device(np.zeros(capacity, dtype=np.float64))
    args = (
        cuda.to_device(group),
        cuda.to_device(value),
        d_table_keys,
        d_table_sums,
        capacity,
        n,
    )
    if launch is None:
        module.run(*args)
    else:
        kernel, grid, block = launch
        kernel[grid, block](*args)
    cuda.synchronize()
    table_keys = d_table_keys.copy_to_host()
    table_sums = d_table_sums.copy_to_host()

    want_keys, inverse = np.unique(group, return_inverse=True)
    want_sums = np.bincount(inverse, weights=value, minlength=want_keys.size)
    scale = np.bincount(inverse, weights=np.abs(value), minlength=want_keys.size)
    occupied = table_keys != -1
    assert np.all(table_sums[~occupied] == 0), "sum in an empty slot"
    got_keys = table_keys[occupied]
    got_sums = table_sums[occupied]
    order = np.argsort(got_keys)
    assert np.array_equal(got_keys[order], want_keys), "group set differs"
    error = np.abs(got_sums[order] - want_sums)
    assert np.all(error <= 1e-9 * scale + 1e-300), f"max error {error.max()}"


def test_groupby_sum() -> None:
    module = load_module()
    kernel = cuda.jit(module.kernel)
    rng = np.random.default_rng(7)
    for n in SIZES:
        value = rng.standard_normal(n) * 100.0
        for groups in sorted({min(n, 1), min(n, 1000), n // 2}):
            if n > 0 and groups == 0:
                continue
            keys = distinct_groups(groups, rng)
            group = keys[rng.integers(0, max(groups, 1), size=n)]
            capacity = table_capacity(groups)
            aggregate_and_check(module, group, value, capacity, None)
            aggregate_and_check(module, np.sort(group), value, capacity, None)
            aggregate_and_check(module, group, value, capacity, (kernel, 3, 100))
            aggregate_and_check(module, group, value, 4 * capacity, (kernel, 2, 1024))
            print(f"groupby_sum n={n} groups={groups}: ok")

    # Groups that all have the home slot capacity - 2, so the probes wrap around.
    groups = 100
    capacity = table_capacity(groups)
    keys = colliding_groups(groups, capacity)
    for n in (1000, 1000003):
        group = keys[rng.integers(0, groups, size=n)]
        value = rng.standard_normal(n)
        aggregate_and_check(module, group, value, capacity, None)
        aggregate_and_check(module, group, value, capacity, (kernel, 1, 32))
        print(f"groupby_sum colliding n={n} groups={groups}: ok")


if __name__ == "__main__":
    test_groupby_sum()
    print("PASS")
