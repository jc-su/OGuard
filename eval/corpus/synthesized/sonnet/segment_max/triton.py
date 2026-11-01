import torch
import triton
import triton.language as tl


@triton.jit
def kernel(offsets, values, maxes, g, BLOCK: tl.constexpr):
    # One program per segment; g is the grid size.
    j = tl.program_id(0).to(tl.int64)
    begin = tl.load(offsets + j).to(tl.int64)
    end = tl.load(offsets + j + 1).to(tl.int64)
    acc = tl.full([BLOCK], -9223372036854775807 - 1, tl.int64)
    for start in range(begin, end, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        v = tl.load(values + idx, mask=idx < end, other=-9223372036854775807 - 1)
        acc = tl.maximum(acc, v)
    tl.store(maxes + j, tl.max(acc, axis=0))


TYPES = {"offsets": "*i32", "values": "*i64", "maxes": "*i64", "g": "i64"}
CONSTANTS = {"BLOCK": 512}
NUM_WARPS = 4


def run(
    offsets: torch.Tensor, values: torch.Tensor, maxes: torch.Tensor, g: int
) -> None:
    if g == 0:
        return
    kernel[(g,)](offsets, values, maxes, g, **CONSTANTS, num_warps=NUM_WARPS)
