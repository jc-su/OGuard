import triton
import triton.language as tl

TYPES = {
    "counts_ptr": "*i64",
    "sums_ptr": "*fp64",
    "codes_out_ptr": "*i32",
    "counts_out_ptr": "*i64",
    "sums_out_ptr": "*fp64",
    "num_groups_ptr": "*i32",
}

CONSTANTS = {"BLOCK_SIZE": 128}

NUM_WARPS = 4


@triton.jit
def kernel(
    counts_ptr,
    sums_ptr,
    codes_out_ptr,
    counts_out_ptr,
    sums_out_ptr,
    num_groups_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    in_range = offs < 100

    cnt = tl.load(counts_ptr + offs, mask=in_range, other=0)
    sm = tl.load(sums_ptr + offs, mask=in_range, other=0.0)
    valid = in_range & (cnt > 0)

    valid_i = valid.to(tl.int32)
    prefix = tl.cumsum(valid_i, 0) - 1
    pos = tl.where(valid, prefix, 0)

    tl.store(codes_out_ptr + pos, offs.to(tl.int32), mask=valid)
    tl.store(counts_out_ptr + pos, cnt, mask=valid)
    tl.store(sums_out_ptr + pos, sm, mask=valid)

    total = tl.sum(valid_i, axis=0)
    tl.store(num_groups_ptr, total)
