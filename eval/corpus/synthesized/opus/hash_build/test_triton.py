"""GPU test of the Triton hash_build kernel against a NumPy reference.

Run from the parent directory: `python -P hash_build/test_triton.py`. The -P
flag keeps this directory off sys.path, so its triton.py and numba.py do not
shadow the packages.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import torch

SIZES = (0, 1, 1000, 1000003)
C1 = np.uint64(0xFF51AFD7ED558CCD)
C2 = np.uint64(0xC4CEB9FE1A85EC53)
S33 = np.uint64(33)


def load_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("triton.py")
    spec = importlib.util.spec_from_file_location("hash_build_triton", path)
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


def random_keys(n: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Distinct keys: an odd multiplier is a bijection on 64-bit integers."""
    base = rng.permutation(n).astype(np.uint64) + np.uint64(12345)
    keys = (base * np.uint64(0x9E3779B97F4A7C15)).view(np.int64)
    return keys[keys != -1]


def colliding_keys(n: int, capacity: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Distinct keys that all have the home slot capacity - 3, so probes wrap."""
    high = rng.permutation(n).astype(np.uint64) + np.uint64(1)
    return fmix64_inverse((high * np.uint64(capacity)) + np.uint64(capacity - 3))


def check_table(
    keys: npt.NDArray[np.int64],
    table_keys: npt.NDArray[np.int64],
    table_rows: npt.NDArray[np.int32],
) -> None:
    n = keys.size
    capacity = table_keys.size
    occupied = table_keys != -1
    assert int(occupied.sum()) == n, "occupied slot count differs from n"
    rows = table_rows[occupied]
    assert np.array_equal(np.sort(rows), np.arange(n, dtype=np.int32)), "row ids"
    assert np.array_equal(keys[rows], table_keys[occupied]), "slot key != keys[row]"
    if n == 0:
        return
    # Linear probing: every slot from the home slot of a key up to the slot
    # of the key is occupied. Rotate the table so it ends at an empty slot;
    # then no run of occupied slots wraps around.
    mask = capacity - 1
    last_empty = int(np.flatnonzero(~occupied)[-1])
    order = (np.arange(capacity) + last_empty + 1) & mask
    rotated_empty = ~occupied[order]
    run_start = np.maximum.accumulate(np.where(rotated_empty, np.arange(capacity), -1))
    slots = np.flatnonzero(occupied)
    home = (fmix64(table_keys[slots]) & np.uint64(mask)).astype(np.int64)
    distance = (slots - home) & mask
    position = (slots - last_empty - 1) & mask
    assert np.all(position - run_start[position] > distance), "probe chain broken"


def build(module: ModuleType, keys: npt.NDArray[np.int64], capacity: int) -> None:
    n = keys.size
    d_keys = torch.from_numpy(keys).cuda()
    d_table_keys = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    d_table_rows = torch.full((capacity,), -7, dtype=torch.int32, device="cuda")
    module.run(d_keys, d_table_keys, d_table_rows, capacity, n)
    torch.cuda.synchronize()
    check_table(keys, d_table_keys.cpu().numpy(), d_table_rows.cpu().numpy())


def test_hash_build() -> None:
    module = load_module()
    rng = np.random.default_rng(5)
    for n in SIZES:
        keys = random_keys(n, rng)
        assert keys.size == n
        build(module, keys, table_capacity(n))
        sequential = np.arange(n, dtype=np.int64) - n // 2
        build(module, sequential[sequential != -1], table_capacity(n) * 4)
        print(f"hash_build n={n}: ok")
    capacity = table_capacity(1000)
    keys = colliding_keys(1000, capacity, rng)
    assert np.unique(keys).size == keys.size and not np.any(keys == -1)
    build(module, keys, capacity)
    print("hash_build colliding n=1000: ok")


if __name__ == "__main__":
    test_hash_build()
    print("PASS")
