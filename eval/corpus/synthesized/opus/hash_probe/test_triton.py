"""GPU test of the Triton hash_probe kernel against a NumPy reference.

Run from the parent directory: `python -P hash_probe/test_triton.py`. The -P
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
C1 = np.uint64(0xFF51AFD7ED558CCD)
C2 = np.uint64(0xC4CEB9FE1A85EC53)
S33 = np.uint64(33)
GUARD = 64
UNSET = -7


class Tile(NamedTuple):
    """A tile shape for a direct launch of the kernel, in place of run()."""

    block: int
    num_warps: int


# One key per lane in four warps, and eight keys per lane in four warps.
TILES = (Tile(128, 4), Tile(1024, 4))


def load_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("triton.py")
    spec = importlib.util.spec_from_file_location("hash_probe_triton", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fmix64(keys: npt.NDArray[np.int64]) -> npt.NDArray[np.uint64]:
    h = keys.view(np.uint64).copy()
    h ^= h >> S33
    h *= C1
    h ^= h >> S33
    h *= C2
    h ^= h >> S33
    return h


def fmix64_inverse(h: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    x = h.copy()
    x ^= x >> S33
    x *= np.uint64(pow(int(C2), -1, 1 << 64))
    x ^= x >> S33
    x *= np.uint64(pow(int(C1), -1, 1 << 64))
    x ^= x >> S33
    return x.view(np.int64)


def table_capacity(n: int) -> int:
    capacity = 1
    while capacity < 2 * n:
        capacity *= 2
    return capacity


def build_table(
    keys: npt.NDArray[np.int64], capacity: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Build a linear probing table with the hash of hash_build on the CPU.

    The keys go in by increasing home slot. Then the slot of the i-th key is
    max(home_i, slot_(i-1) + 1), which is i + max over j <= i of (home_j - j).
    The keys whose slot would be past the table end wrap around: they go
    last, into the first empty slots from slot 0 on.
    """
    table_keys = np.full(capacity, -1, dtype=np.int64)
    table_rows = np.zeros(capacity, dtype=np.int32)
    homes = (fmix64(keys) & np.uint64(capacity - 1)).astype(np.int64)
    order = np.argsort(homes, kind="stable")
    rank = np.arange(keys.size)
    slots = rank + np.maximum.accumulate(homes[order] - rank) if keys.size else rank
    fits = slots < capacity
    table_keys[slots[fits]] = keys[order[fits]]
    table_rows[slots[fits]] = order[fits]
    wrapped = np.flatnonzero(table_keys == -1)[: int((~fits).sum())]
    table_keys[wrapped] = keys[order[~fits]]
    table_rows[wrapped] = order[~fits]
    return table_keys, table_rows


def expected_pairs(
    build_keys: npt.NDArray[np.int64], probe_keys: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Return the (build row, probe row) matches, ordered by probe row."""
    if build_keys.size == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    order = np.argsort(build_keys, kind="stable")
    sorted_keys = build_keys[order]
    index = np.minimum(np.searchsorted(sorted_keys, probe_keys), build_keys.size - 1)
    hit = sorted_keys[index] == probe_keys
    probe_rows = np.flatnonzero(hit).astype(np.int32)
    return order[index[hit]].astype(np.int32), probe_rows


def probe_and_check(
    module: ModuleType,
    build_keys: npt.NDArray[np.int64],
    probe_keys: npt.NDArray[np.int64],
    out_cap: int,
    tile: Tile | None,
) -> None:
    """Probe with module.run, or with a direct launch of the kernel with the given tile."""
    m = probe_keys.size
    capacity = table_capacity(build_keys.size)
    table_keys, table_rows = build_table(build_keys, capacity)
    # The outputs are the first out_cap elements of larger buffers, so a
    # write past out_cap lands in the guard elements.
    d_out_build = torch.full((out_cap + GUARD,), UNSET, dtype=torch.int32, device="cuda")
    d_out_probe = torch.full((out_cap + GUARD,), UNSET, dtype=torch.int32, device="cuda")
    d_out_count = torch.zeros(1, dtype=torch.int32, device="cuda")
    args = (
        torch.from_numpy(probe_keys).cuda(),
        torch.from_numpy(table_keys).cuda(),
        torch.from_numpy(table_rows).cuda(),
        capacity,
        d_out_build[:out_cap],
        d_out_probe[:out_cap],
        d_out_count,
        m,
        out_cap,
    )
    if tile is None:
        module.run(*args)
    elif m > 0:
        grid = ((m + tile.block - 1) // tile.block,)
        module.kernel[grid](*args, BLOCK=tile.block, num_warps=tile.num_warps)
    torch.cuda.synchronize()

    want_build, want_probe = expected_pairs(build_keys, probe_keys)
    count = int(d_out_count.item())
    assert count == want_probe.size, f"out_count {count} != {want_probe.size}"
    written = min(count, out_cap)
    out_build = d_out_build.cpu().numpy()
    out_probe = d_out_probe.cpu().numpy()
    assert np.all(out_build[written:] == UNSET), "write past the pairs"
    assert np.all(out_probe[written:] == UNSET), "write past the pairs"
    got_probe = out_probe[:written]
    got_build = out_build[:written]
    order = np.argsort(got_probe, kind="stable")
    got_probe = got_probe[order]
    got_build = got_build[order]
    assert np.all(np.diff(got_probe) > 0), "a probe row appears twice"
    index = np.searchsorted(want_probe, got_probe)
    assert np.array_equal(want_probe[index], got_probe), "pair with no match"
    assert np.array_equal(want_build[index], got_build), "wrong build row"


def distinct_keys(n: int, offset: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Distinct keys: an odd multiplier is a bijection on 64-bit integers."""
    base = rng.permutation(n).astype(np.uint64) + np.uint64(offset)
    return (base * np.uint64(0x9E3779B97F4A7C15)).view(np.int64)


def probe_keys_for(
    build_keys: npt.NDArray[np.int64], m: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Probe row 0 and about half the others repeat build keys; one is -1."""
    misses = distinct_keys(m, 1 << 40, rng)
    probe = misses.copy()
    if build_keys.size > 0:
        take = rng.random(m) < 0.5
        take[0] = True
        probe[take] = rng.choice(build_keys, size=int(take.sum()))
    if m >= 3:
        probe[m // 3] = -1
    return probe


def test_hash_probe() -> None:
    module = load_module()
    rng = np.random.default_rng(6)
    for n in SIZES:
        build_keys = distinct_keys(n, 12345, rng)
        probe_keys = probe_keys_for(build_keys, n, rng)
        matches = expected_pairs(build_keys, probe_keys)[1].size
        probe_and_check(module, build_keys, probe_keys, n, None)
        probe_and_check(module, build_keys, probe_keys, matches // 2, None)
        probe_and_check(module, build_keys, probe_keys, n, TILES[0])
        probe_and_check(module, build_keys, probe_keys, matches // 3, TILES[1])
        print(f"hash_probe n={n} matches={matches}: ok")

    # Keys that all have the home slot capacity - 3, so the probes wrap around.
    n = 1000
    capacity = table_capacity(n)
    high = rng.permutation(2 * n).astype(np.uint64) + np.uint64(1)
    keys = fmix64_inverse(high * np.uint64(capacity) + np.uint64(capacity - 3))
    assert not np.any(keys == -1)
    build_keys = keys[:n]
    probe_keys = rng.permutation(keys)
    probe_and_check(module, build_keys, probe_keys, 2 * n, None)
    probe_and_check(module, build_keys, probe_keys, 2 * n, TILES[0])
    print(f"hash_probe colliding n={n}: ok")


if __name__ == "__main__":
    test_hash_probe()
    print("PASS")
