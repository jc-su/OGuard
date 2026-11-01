"""GPU test of histogram/numba.py against a NumPy reference.

Run from the parent directory with `python -P histogram/test_numba.py`.
The -P flag keeps this directory off sys.path, so numba.py in this directory
does not shadow the numba package.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
from numba import cuda

SIZES: tuple[int, ...] = (0, 1, 1000, 1000003)
SHIFTS: tuple[int, ...] = (0, 3, 8, 13, 16, 24)
BINS: int = 256


def load_module() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("histogram_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key_sets(n: int, rng: np.random.Generator) -> dict[str, npt.NDArray[np.uint32]]:
    """Uniform keys, and two skewed key sets that put many keys in one bin."""
    return {
        "uniform": rng.integers(0, 2**32, n, dtype=np.uint32),
        "constant": np.full(n, 0xA5C3E17F, dtype=np.uint32),
        "four_values": rng.choice(
            np.array([0, 0xFFFFFFFF, 0x12345678, 0x80000000], dtype=np.uint32), n
        ),
    }


def check(module: ModuleType, name: str, keys: npt.NDArray[np.uint32], shift: int) -> None:
    n = keys.size
    digits = ((keys >> np.uint32(shift)) & np.uint32(0xFF)).astype(np.int64)
    expected = np.bincount(digits, minlength=BINS).astype(np.int32)
    d_keys = cuda.to_device(keys)
    d_counts = cuda.to_device(np.zeros(BINS, dtype=np.int32))
    module.run(d_keys, d_counts, shift, n)
    got = d_counts.copy_to_host()
    bad = np.flatnonzero(got != expected)
    if bad.size != 0:
        b = int(bad[0])
        raise AssertionError(
            f"n={n} keys={name} shift={shift}: {bad.size} bins differ, first bin {b}: "
            f"got {got[b]}, expected {expected[b]}"
        )


def main() -> None:
    module = load_module()
    rng = np.random.default_rng(11)
    for n in SIZES:
        for name, keys in key_sets(n, rng).items():
            for shift in SHIFTS:
                check(module, name, keys, shift)
        print(f"histogram numba n={n} key_sets=3 shifts={list(SHIFTS)}: pass")


if __name__ == "__main__":
    main()
