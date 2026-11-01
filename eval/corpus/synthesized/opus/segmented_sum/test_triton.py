"""GPU test of segmented_sum/triton.py against a NumPy reference.

Run from the parent directory with `python -P segmented_sum/test_triton.py`.
The -P flag keeps this directory off sys.path, so triton.py and numba.py in
this directory do not shadow the packages.

The values are multiples of 1/8 with magnitude at most 2**17, so every partial
sum is exact in float64 and the check can require equality, independent of
the order of the additions.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import torch

SIZES: tuple[int, ...] = (0, 1, 1000, 1000003)
SMALL_GRID: int = 3  # Programs of a direct launch, to exercise the grid-stride loops.


def load_module() -> ModuleType:
    path = Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("segmented_sum_triton", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_offsets(n: int, g: int, rng: np.random.Generator) -> npt.NDArray[np.int32]:
    """Nondecreasing offsets with offsets[g] == n. offsets[0] can be above 0, and segments can be empty."""
    offsets = np.sort(rng.integers(0, n + 1, g + 1))
    offsets[-1] = n
    return offsets.astype(np.int32)


def skewed_offsets(n: int, rng: np.random.Generator) -> npt.NDArray[np.int32]:
    """n // 4 short segments (0 to 3 values) and three long segments.

    The kernel then uses row mode, and sums the long segments as spans; one
    of them is the last segment of its tile. Requires n >= 1000.
    """
    g = n // 4
    lengths = rng.integers(0, 4, g)
    long_rows = np.array([31, g // 2, g - 1])
    lengths[long_rows] = 0
    remainder = n - int(lengths.sum())
    lengths[long_rows] = (remainder // 3, remainder // 3, remainder - 2 * (remainder // 3))
    offsets = np.zeros(g + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return offsets.astype(np.int32)


def offset_cases(n: int, rng: np.random.Generator) -> dict[str, npt.NDArray[np.int32]]:
    """Cases that select each mapping and each span width of the kernel at n = 1000003."""
    cases = {
        "one_segment": np.array([0, n], dtype=np.int32),
        "late_start": np.array([max(0, n - 40), n], dtype=np.int32),
        "g=n/5000": random_offsets(n, max(1, n // 5000), rng),
        "g=n/100": random_offsets(n, max(1, n // 100), rng),
        "g=n/10": random_offsets(n, max(1, n // 10), rng),
        "g=n+5": random_offsets(n, n + 5, rng),
    }
    if n >= 1000:
        cases["skewed"] = skewed_offsets(n, rng)
    if n == 0:
        cases["g=0"] = np.zeros(1, dtype=np.int32)
    return cases


def check(
    module: ModuleType,
    name: str,
    offsets: npt.NDArray[np.int32],
    values: npt.NDArray[np.float64],
    small_grid: bool,
) -> None:
    """Run module.run, or a direct launch with SMALL_GRID programs, and compare."""
    g = offsets.size - 1
    n = values.size
    segment = np.repeat(np.arange(g), np.diff(offsets))
    expected = np.bincount(segment, weights=values[offsets[0] :], minlength=g)
    d_offsets = torch.from_numpy(offsets).cuda()
    d_values = torch.from_numpy(values).cuda()
    # Fill the output with NaN so that an unwritten segment fails the check.
    d_sums = torch.full((g,), np.nan, dtype=torch.float64, device="cuda")
    if not small_grid:
        module.run(d_offsets, d_values, d_sums, g)
    elif g > 0:
        module.kernel[(SMALL_GRID,)](
            d_offsets, d_values, d_sums, g, **module.CONSTANTS, num_warps=module.NUM_WARPS
        )
    got = d_sums.cpu().numpy()
    bad = np.flatnonzero(got != expected)
    if bad.size != 0:
        j = int(bad[0])
        raise AssertionError(
            f"n={n} case={name} g={g} small_grid={small_grid}: {bad.size} mismatches, "
            f"first at {j}: got {got[j]}, expected {expected[j]}"
        )


def main() -> None:
    module = load_module()
    rng = np.random.default_rng(12)
    for n in SIZES:
        values = rng.integers(-(2**20), 2**20, n) / 8.0
        cases = offset_cases(n, rng)
        for name, offsets in cases.items():
            check(module, name, offsets, values, small_grid=False)
            check(module, name, offsets, values, small_grid=True)
        print(f"segmented_sum triton n={n} cases={list(cases)}: pass")


if __name__ == "__main__":
    main()
