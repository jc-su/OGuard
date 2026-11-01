import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(valid, rows, count, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = row < n

    byte_idx = row // 8
    bit_idx = row % 8
    byte = tl.load(valid + byte_idx, mask=in_range, other=0)
    bit = (byte >> bit_idx) & 1
    is_valid = in_range & (bit != 0)

    valid_i32 = is_valid.to(tl.int32)
    inclusive = tl.cumsum(valid_i32, axis=0)
    exclusive = inclusive - valid_i32
    block_total = tl.sum(valid_i32, axis=0)

    # Reserve a contiguous run of `block_total` slots for this block in one
    # atomic instead of one atomicAdd per valid row.
    base = tl.atomic_add(count, block_total)
    dest = base + exclusive
    tl.store(rows + dest, row.to(tl.int32), mask=is_valid)


TYPES = {
    "valid": "*u8",
    "rows": "*i32",
    "count": "*i32",
    "n": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE}
NUM_WARPS = 8


def run(valid, rows, count, n):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](valid, rows, count, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
