"""Hash partitioning of int64 keys with partition sizes, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n", "p"])
def kernel(keys, part, counts, n, p, BLOCK: tl.constexpr, TILES: tl.constexpr, BINS: tl.constexpr):
    # Each program handles TILES consecutive tiles of BLOCK rows, accumulates
    # the partition sizes of its rows in registers, and adds the nonzero sizes
    # to counts with one atomic per partition. BINS is the largest p.
    first = tl.program_id(0).to(tl.int64) * (TILES * BLOCK)
    partitions = p.to(tl.uint32)
    sizes = tl.zeros([BINS], dtype=tl.int32)
    for t in range(TILES):
        rows = first + t * BLOCK + tl.arange(0, BLOCK)
        in_range = rows < n
        key = tl.load(keys + rows, mask=in_range, other=0).to(tl.uint64, bitcast=True)
        hashed = ((key * 0x9E3779B97F4A7C15) >> 32).to(tl.uint32)
        target = (hashed % partitions).to(tl.int32)
        tl.store(part + rows, target, mask=in_range)
        sizes += tl.histogram(target, BINS, mask=in_range)
    bins = tl.arange(0, BINS)
    tl.atomic_add(counts + bins, sizes, mask=sizes > 0)


TYPES: dict[str, str] = {
    "keys": "*i64",
    "part": "*i32",
    "counts": "*i32",
    "n": "i64",
    "p": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024, "TILES": 8, "BINS": 1024}
NUM_WARPS: int = 4


def run(
    keys: torch.Tensor,
    part: torch.Tensor,
    counts: torch.Tensor,
    n: int,
    p: int,
) -> None:
    if not 1 <= p <= CONSTANTS["BINS"]:
        raise ValueError(f"p must be in [1, {CONSTANTS['BINS']}], got {p}")
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"] * CONSTANTS["TILES"]),)
    kernel[grid](keys, part, counts, n, p, **CONSTANTS, num_warps=NUM_WARPS)
