"""Task 11, histogram: count the keys by the byte (key >> shift) & 255.

This is the counting pass of one radix sort digit. The grid holds at most
PROGRAMS_PER_SM programs per SM. Each program loops over tiles of BLOCK keys
with a grid stride and counts them in a private histogram of BINS bins, then
adds its nonzero bins to counts with atomic adds. counts must be zero on
entry.

The scalar annotations (tl.int64) set the Triton types of the scalar
arguments at launch. Without them, Triton types a small Python int as i32.
ty does not accept a Triton dtype in an annotation, and reports the calls of
the Triton library functions that are @triton.jit functions (tl.zeros) as
argument errors. The lines where this occurs carry `ty: ignore` comments. A
launch takes host values (tensors, Python scalars, launch options), which the
device-side annotations do not describe, so run() casts the launcher to
Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

BLOCK = 2048  # Keys per tile.
BINS = 256  # One bin per byte value.
NUM_WARPS = 4
PROGRAMS_PER_SM = 4

TYPES = {
    "keys": "*u32",
    "counts": "*i32",
    "shift": "i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK, "BINS": BINS}


@triton.jit
def kernel(
    keys: tl.tensor,
    counts: tl.tensor,
    shift: tl.int64,  # ty: ignore[invalid-type-form]
    n: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
) -> None:
    local = tl.zeros([BINS], tl.int32)  # ty: ignore[invalid-argument-type]
    first = tl.program_id(0).to(tl.int64) * BLOCK
    stride = tl.num_programs(0).to(tl.int64) * BLOCK
    for start in tl.range(first, n, stride):
        rows = start + tl.arange(0, BLOCK)
        in_range = rows < n
        key = tl.load(keys + rows, mask=in_range, other=0)
        digit = ((key >> shift.to(tl.uint32)) & (BINS - 1)).to(tl.int32)
        local += tl.histogram(digit, BINS, mask=in_range)
    tl.atomic_add(counts + tl.arange(0, BINS), local, mask=local != 0, sem="relaxed")


def check_array(name: str, array: torch.Tensor, dtype: torch.dtype, length: int) -> None:
    """Raise ValueError unless array is a contiguous CUDA tensor of dtype and shape (length,)."""
    if (
        array.dtype != dtype
        or tuple(array.shape) != (length,)
        or not array.is_cuda
        or not array.is_contiguous()
    ):
        raise ValueError(
            f"{name}: expected a contiguous CUDA tensor of {dtype} with shape ({length},), "
            f"got {array.dtype} with shape {tuple(array.shape)} on {array.device}"
        )


def run(
    keys: torch.Tensor,
    counts: torch.Tensor,
    shift: int,
    n: int,
) -> None:
    """Add the digit counts of keys[0:n] to counts, which is zero on entry."""
    check_array("keys", keys, torch.uint32, n)
    check_array("counts", counts, torch.int32, BINS)
    if not 0 <= shift <= 24:
        raise ValueError(f"shift {shift} is not in [0, 24]")
    if n == 0:
        return
    sms = torch.cuda.get_device_properties(counts.device).multi_processor_count
    grid = (min((n + BLOCK - 1) // BLOCK, PROGRAMS_PER_SM * sms),)
    launch = cast(Callable[..., object], kernel[grid])
    launch(keys, counts, shift, n, BLOCK=BLOCK, BINS=BINS, num_warps=NUM_WARPS)
