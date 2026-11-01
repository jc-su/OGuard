"""Null count of an Arrow validity bitmap, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n"])
def kernel(valid, count, n, BLOCK: tl.constexpr):
    # One lane per bitmap byte. The last byte can hold fewer than 8 rows.
    byte = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    rows = tl.minimum(tl.maximum(n - byte * 8, 0), 8).to(tl.int32)
    bits = tl.load(valid + byte, mask=rows > 0, other=0).to(tl.int32)
    bits &= (1 << rows) - 1
    bits = bits - ((bits >> 1) & 0x55)
    bits = (bits & 0x33) + ((bits >> 2) & 0x33)
    bits = (bits + (bits >> 4)) & 0x0F
    nulls = tl.sum(rows - bits, axis=0)
    tl.atomic_add(count, nulls.to(tl.int64))


TYPES: dict[str, str] = {"valid": "*u8", "count": "*i64", "n": "i64"}
CONSTANTS: dict[str, int] = {"BLOCK": 4096}
NUM_WARPS: int = 4


def run(valid: torch.Tensor, count: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(triton.cdiv(n, 8), CONSTANTS["BLOCK"]),)
    kernel[grid](valid, count, n, **CONSTANTS, num_warps=NUM_WARPS)
