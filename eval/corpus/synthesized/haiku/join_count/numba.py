from numba import cuda
import numpy as np


def kernel(build, probe, counts, b, p):
    """Count occurrences of probe elements in build array."""
    idx = cuda.grid(1)
    if idx < p:
        probe_val = probe[idx]
        count = 0
        # Binary search in sorted build array
        lo, hi = 0, b
        while lo < hi:
            mid = (lo + hi) // 2
            if build[mid] < probe_val:
                lo = mid + 1
            else:
                hi = mid

        # Count equal elements
        while lo < b and build[lo] == probe_val:
            count += 1
            lo += 1

        counts[idx] = count


TYPES = (
    cuda.int64[:],
    cuda.int64[:],
    cuda.int32[:],
    cuda.int64,
    cuda.int64,
)


def run(build, probe, counts, b, p, grid_size=None):
    """Launch the join_count kernel."""
    if grid_size is None:
        grid_size = max(1, (p + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](build, probe, counts, b, p)
