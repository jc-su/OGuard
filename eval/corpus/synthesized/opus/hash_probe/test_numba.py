"""GPU test of the Numba hash_probe kernel against a NumPy reference.

Run from the parent directory: `python -P hash_probe/test_numba.py`. The -P flag
keeps this directory off sys.path, so its numba.py does not shadow Numba.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
from numba import cuda, njit
from numba.cuda.dispatcher import CUDADispatcher

SIZES = (0, 1, 1000, 1000003)
C1 = np.uint64(0xFF51AFD7ED558CCD)
C2 = np.uint64(0xC4CEB9FE1A85EC53)
S33 = np.uint64(33)
GUARD = 64
UNSET = -7


def load_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("numba.py")
    spec = importlib.util.spec_from_file_location("hash_probe_numba", path)
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


@njit
def insert_all(
    keys: npt.NDArray[np.int64],
    homes: npt.NDArray[np.int64],
    table_keys: npt.NDArray[np.int64],
    table_rows: npt.NDArray[np.int32],
) -> None:
    mask = table_keys.size - 1
    for i in range(keys.size):
        slot = homes[i]
        while table_keys[slot] != -1:
            slot = (slot + 1) & mask
        table_keys[slot] = keys[i]
        table_rows[slot] = i


def build_table(
    keys: npt.NDArray[np.int64], capacity: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Build the hash_build table on the CPU."""
    table_keys = np.full(capacity, -1, dtype=np.int64)
    table_rows = np.zeros(capacity, dtype=np.int32)
    homes = (fmix64(keys) & np.uint64(capacity - 1)).astype(np.int64)
    insert_all(keys, homes, table_keys, table_rows)
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
    launch: tuple[CUDADispatcher, int, int] | None,
) -> None:
    """Probe with module.run, or with the given kernel, grid, and block size."""
    m = probe_keys.size
    capacity = table_capacity(build_keys.size)
    table_keys, table_rows = build_table(build_keys, capacity)
    d_out_build = cuda.to_device(np.full(out_cap + GUARD, UNSET, dtype=np.int32))
    d_out_probe = cuda.to_device(np.full(out_cap + GUARD, UNSET, dtype=np.int32))
    d_out_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    args = (
        cuda.to_device(probe_keys),
        cuda.to_device(table_keys),
        cuda.to_device(table_rows),
        capacity,
        d_out_build,
        d_out_probe,
        d_out_count,
        m,
        out_cap,
    )
    if launch is None:
        module.run(*args)
    else:
        kernel, grid, block = launch
        kernel[grid, block](*args)
    cuda.synchronize()

    want_build, want_probe = expected_pairs(build_keys, probe_keys)
    count = int(d_out_count.copy_to_host()[0])
    assert count == want_probe.size, f"out_count {count} != {want_probe.size}"
    written = min(count, out_cap)
    out_build = d_out_build.copy_to_host()
    out_probe = d_out_probe.copy_to_host()
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
    kernel = cuda.jit(module.kernel)
    rng = np.random.default_rng(6)
    for n in SIZES:
        build_keys = distinct_keys(n, 12345, rng)
        probe_keys = probe_keys_for(build_keys, n, rng)
        matches = expected_pairs(build_keys, probe_keys)[1].size
        probe_and_check(module, build_keys, probe_keys, n, None)
        probe_and_check(module, build_keys, probe_keys, matches // 2, None)
        probe_and_check(module, build_keys, probe_keys, n, (kernel, 3, 100))
        probe_and_check(module, build_keys, probe_keys, matches // 3, (kernel, 2, 1024))
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
    probe_and_check(module, build_keys, probe_keys, 2 * n, (kernel, 1, 32))
    print(f"hash_probe colliding n={n}: ok")


if __name__ == "__main__":
    test_hash_probe()
    print("PASS")
