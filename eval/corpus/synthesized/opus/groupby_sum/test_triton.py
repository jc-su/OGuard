"""GPU test of the Triton groupby_sum kernel against a NumPy reference.

Run from the parent directory: `python -P groupby_sum/test_triton.py`. The -P
flag keeps this directory off sys.path, so its triton.py and numba.py do not
shadow the packages.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch

SIZES = (0, 1, 1000, 1000003)
U32 = (1 << 32) - 1


class Tile(NamedTuple):
    """A tile shape for a direct launch of the kernel, in place of run()."""

    block: int
    num_warps: int


# One row per lane in four warps, and four rows per lane in eight warps.
TILES = (Tile(128, 4), Tile(1024, 8))


def load_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("triton.py")
    spec = importlib.util.spec_from_file_location("groupby_sum_triton", path)
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
    groups = [fmix32_inverse(((i + 1) << 16) | (capacity - 2)) for i in range(count)]
    assert -1 not in groups
    assert all(fmix32(g) & (capacity - 1) == capacity - 2 for g in groups)
    return np.array(groups, dtype=np.int32)


def interleaved_groups(count: int, block: int) -> npt.NDArray[np.int32]:
    """Distinct groups whose hashes differ only in the low log2(block) bits.

    The kernel sorts the rows of a tile by the hash with these bits replaced
    by the lane, so the sort orders the rows of these groups by lane only and
    interleaves them.
    """
    assert count <= block
    groups = [fmix32_inverse((0x5A5A5 * block) | i) for i in range(count)]
    assert -1 not in groups
    return np.array(groups, dtype=np.int32)


def aggregate_and_check(
    module: ModuleType,
    group: npt.NDArray[np.int32],
    value: npt.NDArray[np.float64],
    capacity: int,
    tile: Tile | None,
) -> None:
    """Aggregate with module.run, or with a direct launch of the kernel with the given tile."""
    n = group.size
    d_table_keys = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
    d_table_sums = torch.zeros(capacity, dtype=torch.float64, device="cuda")
    args = (
        torch.from_numpy(group).cuda(),
        torch.from_numpy(value).cuda(),
        d_table_keys,
        d_table_sums,
        capacity,
        n,
    )
    if tile is None:
        module.run(*args)
    elif n > 0:
        grid = ((n + tile.block - 1) // tile.block,)
        module.kernel[grid](*args, BLOCK=tile.block, num_warps=tile.num_warps)
    torch.cuda.synchronize()
    table_keys = d_table_keys.cpu().numpy()
    table_sums = d_table_sums.cpu().numpy()

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
            aggregate_and_check(module, group, value, capacity, TILES[0])
            aggregate_and_check(module, group, value, 4 * capacity, TILES[1])
            print(f"groupby_sum n={n} groups={groups}: ok")

    # Groups that all have the home slot capacity - 2, so the probes wrap around.
    groups = 100
    capacity = table_capacity(groups)
    keys = colliding_groups(groups, capacity)
    for n in (1000, 1000003):
        group = keys[rng.integers(0, groups, size=n)]
        value = rng.standard_normal(n)
        aggregate_and_check(module, group, value, capacity, None)
        aggregate_and_check(module, group, value, capacity, TILES[0])
        print(f"groupby_sum colliding n={n} groups={groups}: ok")

    # Groups whose rows the sort of a tile interleaves.
    keys = interleaved_groups(groups, module.BLOCK)
    for n in (1000, 1000003):
        group = keys[rng.integers(0, groups, size=n)]
        value = rng.standard_normal(n)
        aggregate_and_check(module, group, value, table_capacity(groups), None)
        print(f"groupby_sum interleaved n={n} groups={groups}: ok")


if __name__ == "__main__":
    test_groupby_sum()
    print("PASS")
