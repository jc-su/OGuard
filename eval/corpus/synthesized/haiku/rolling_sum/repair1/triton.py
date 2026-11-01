import triton
import triton.language as tl
import torch


@triton.jit
def kernel(values, out, n, window, BLOCK_SIZE: tl.constexpr):
    """
    Compute rolling sum with fixed window size.
    values: float64 array of n values
    out: float64 output array of n values
    n: number of values
    window: window size (1 to 1024)
    """
    idx = tl.program_id(0) * BLOCK_SIZE

    for i in range(BLOCK_SIZE):
        pos = idx + i
        if pos < n:
            # Compute sum of values[max(0, pos - window + 1):pos + 1]
            total = tl.cast(0.0, tl.float64)
            start = tl.maximum(0, pos - window + 1)
            for j in range(start, pos + 1):
                # Ensure j is within bounds
                if j >= 0 and j < n:
                    value = tl.load(values + j)
                    total = total + value

            tl.store(out + pos, total, eviction_policy='evict_last')


TYPES = {
    "values": "*fp64",
    "out": "*fp64",
    "n": "i64",
    "window": "i64"
}
CONSTANTS = {"BLOCK_SIZE": 256}
NUM_WARPS = 4


def run(values, out, n, window):
    """
    values: torch.tensor(shape=(n,), dtype=torch.float64)
    out: torch.tensor(shape=(n,), dtype=torch.float64)
    n: int (number of values)
    window: int (window size, 1 to 1024)
    """
    grid = ((n + CONSTANTS["BLOCK_SIZE"] - 1) // CONSTANTS["BLOCK_SIZE"],)
    kernel[grid](values, out, n, window, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
