"""Offsets of a string or list column from its lengths, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n"])
def kernel(lengths, offsets, n, BLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    # The programs split the tiles of BLOCK rows into contiguous ranges. Each
    # program first sums all rows before its range, then scans its range tile
    # by tile and carries the running total. The programs do not communicate,
    # so the scan works for any grid size. The int32 sums wrap the same way as
    # a sequential int32 scan.
    pid = tl.program_id(0).to(tl.int64)
    programs = tl.num_programs(0).to(tl.int64)
    tiles = tl.cdiv(n.to(tl.int64), BLOCK)
    begin = (pid * tiles // programs) * BLOCK
    end = tl.minimum(((pid + 1) * tiles // programs) * BLOCK, n)
    before = tl.zeros([RBLOCK], dtype=tl.int32)
    for start in range(0, begin, RBLOCK):
        rows = start + tl.arange(0, RBLOCK)
        before += tl.load(lengths + rows, mask=rows < begin, other=0)
    total = tl.sum(before, axis=0)
    if pid == 0:
        tl.store(offsets, 0)
    for start in range(begin, end, BLOCK):
        rows = start + tl.arange(0, BLOCK)
        in_range = rows < end
        x = tl.load(lengths + rows, mask=in_range, other=0)
        tl.store(offsets + rows + 1, total + tl.cumsum(x, axis=0), mask=in_range)
        total += tl.sum(x, axis=0)


TYPES: dict[str, str] = {"lengths": "*i32", "offsets": "*i32", "n": "i64"}
CONSTANTS: dict[str, int] = {"BLOCK": 4096, "RBLOCK": 8192}
NUM_WARPS: int = 8
# Upper bound on the number of programs. Each program rereads the rows before
# its range, so the total read volume grows with the number of programs.
PROGRAMS: int = 32


def run(lengths: torch.Tensor, offsets: torch.Tensor, n: int) -> None:
    # At least one program, which writes offsets[0] when n is 0.
    programs = max(1, min(PROGRAMS, triton.cdiv(n, CONSTANTS["BLOCK"])))
    kernel[(programs,)](lengths, offsets, n, **CONSTANTS, num_warps=NUM_WARPS)
