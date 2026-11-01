"""Minimum and maximum of a float64 column (no NaN).

A persistent grid of at most PROGRAMS_PER_SM programs per SM. Each program walks
the column with a grid stride, keeps BLOCK running minima and maxima in
registers, reduces them once, and merges its result into result[0] and
result[1] with one atomic min and one atomic max. With n = 0 no program runs and
the result keeps its +inf and -inf.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, result, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64) * BLOCK
    lane = tl.arange(0, BLOCK)
    low = tl.full((BLOCK,), float("inf"), tl.float64)
    high = tl.full((BLOCK,), float("-inf"), tl.float64)
    for start in range(pid * BLOCK, n, stride):
        offs = start + lane
        in_range = offs < n
        v = tl.load(values + offs, mask=in_range, other=0.0)
        low = tl.minimum(low, tl.where(in_range, v, float("inf")))
        high = tl.maximum(high, tl.where(in_range, v, float("-inf")))
    tl.atomic_min(result, tl.min(low, axis=0), sem="relaxed")
    tl.atomic_max(result + 1, tl.max(high, axis=0), sem="relaxed")


TYPES = {"values": "*fp64", "result": "*fp64", "n": "i64"}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 8
PROGRAMS_PER_SM = 4

_sm_count: dict[int, int] = {}


def run(values: torch.Tensor, result: torch.Tensor, n: int) -> None:
    device = values.device.index
    if device not in _sm_count:
        _sm_count[device] = torch.cuda.get_device_properties(device).multi_processor_count
    programs = min(triton.cdiv(n, CONSTANTS["BLOCK"]), PROGRAMS_PER_SM * _sm_count[device])
    kernel[(programs,)](values, result, n, num_warps=NUM_WARPS, **CONSTANTS)
