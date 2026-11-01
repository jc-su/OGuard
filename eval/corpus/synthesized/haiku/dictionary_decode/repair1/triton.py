import triton
import triton.language as tl
import torch


@triton.jit
def kernel(codes, dictionary, out, n, d, BLOCK_SIZE: tl.constexpr):
    """
    Decode dictionary-encoded values.
    codes: int32 array of n values (each in [0, d))
    dictionary: float64 array of d values
    out: float64 output array of n values
    """
    pid = tl.program_id(0)

    # Process BLOCK_SIZE elements per block
    for i in range(BLOCK_SIZE):
        idx = pid * BLOCK_SIZE + i
        if idx < n:
            # Load code
            code = tl.load(codes + idx, eviction_policy='evict_last')
            # Clamp code to valid range using min/max to ensure provably in-bounds access
            code = tl.minimum(tl.maximum(code, 0), d - 1)
            # Look up in dictionary
            value = tl.load(dictionary + code, eviction_policy='evict_last')
            # Store result
            tl.store(out + idx, value, eviction_policy='evict_last')


TYPES = {
    "codes": "*i32",
    "dictionary": "*fp64",
    "out": "*fp64",
    "n": "i64",
    "d": "i64"
}
CONSTANTS = {"BLOCK_SIZE": 256}
NUM_WARPS = 4


def run(codes, dictionary, out, n, d):
    """
    codes: torch.tensor(shape=(n,), dtype=torch.int32)
    dictionary: torch.tensor(shape=(d,), dtype=torch.float64)
    out: torch.tensor(shape=(n,), dtype=torch.float64)
    n: int (number of codes)
    d: int (dictionary size)
    """
    grid = ((n + CONSTANTS["BLOCK_SIZE"] - 1) // CONSTANTS["BLOCK_SIZE"],)
    kernel[grid](codes, dictionary, out, n, d, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
