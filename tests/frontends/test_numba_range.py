"""The Numba front end's range lowering iterates as Python's range.

The front end replaces Numba's lowering of range iteration so that each
iteration tests the position against the bound (design decision 25). One
GPU thread runs `for i in range(start, stop, step)` and records every value;
the values and their count are Python's, for positive and negative steps,
empty ranges, and ranges whose next value would pass the largest or least
64-bit integer.
"""

import numpy as np
import pytest
from numba import cuda

from opgate.frontends.numba_cuda import install_ranges

LIMIT = 16
CASES = [
    (0, 10, 3),
    (10, 0, -3),
    (5, 5, 1),
    (0, 10, -1),
    (-5, 5, 2),
    (0, 7, 7),
    (3, -4, -1),
    (2**62, 2**63 - 1, 2**61),
    (2**63 - 5, 2**63 - 1, 3),
    (-(2**63) + 10, -(2**63) + 1, -4),
    (-(2**63) + 5, 2**63 - 1, 2**62),
    (-(2**63) + 3, -(2**63), -1),
    (-(2**63), -(2**63) + 1, -1),
    (-(2**63), -(2**63) + 3, 1),
    (2**63 - 1, -(2**63), -(2**63)),
    (0, -(2**63), -(2**63)),
]


def iterate(out, count, start, stop, step):
    k = 0
    for i in range(start, stop, step):
        if k >= out.size:
            break
        out[k] = i
        k += 1
    count[0] = k


@pytest.mark.gpu
@pytest.mark.parametrize(("start", "stop", "step"), CASES)
def test_a_range_yields_the_values_of_python(start: int, stop: int, step: int) -> None:
    _check_range(start, stop, step, np.int64)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("start", "stop", "step", "dtype"),
    [
        (-(2**31) + 3, -(2**31), -1, np.int32),
        (-(2**31), -(2**31) + 1, -1, np.int32),
        (2**31 - 1, -(2**31), -(2**31), np.int32),
        (2**64 - 5, 2**64 - 1, 2, np.uint64),
        (2**64 - 5, 2**64 - 1, 3, np.uint64),
        (1, 0, 1, np.uint64),
    ],
)
def test_ranges_at_other_integer_widths(start, stop, step, dtype) -> None:
    _check_range(start, stop, step, dtype)


def _check_range(start, stop, step, dtype) -> None:
    install_ranges()
    kernel = cuda.jit(iterate)
    out = cuda.to_device(np.zeros(LIMIT, dtype=dtype))
    count = cuda.to_device(np.zeros(1, dtype=np.int64))
    kernel[1, 1](out, count, dtype(start), dtype(stop), dtype(step))
    expected = list(range(start, stop, step))
    assert int(count.copy_to_host()[0]) == len(expected)
    assert out.copy_to_host()[: len(expected)].tolist() == expected[:LIMIT]
