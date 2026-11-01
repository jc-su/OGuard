import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(valid, rows, count, n, BLOCK_SIZE: tl.constexpr):
    # Compute the row index in 64-bit arithmetic: program_id and arange are
    # 32-bit by default, and for an adversarial (oversized) grid
    # `pid * BLOCK_SIZE` can overflow a 32-bit accumulator, defeating the
    # `row < n` mask.
    pid = tl.program_id(0).to(tl.int64)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    in_range = row < n

    # `valid` holds (n + 7) // 8 bytes (signature.json's declared length
    # for this argument). `row < n` alone implies
    # `row // 8 < (n + 7) // 8` mathematically, but a static prover that
    # treats integer division opaquely cannot chain that implication on
    # its own. Guard the byte index against that exact expression too, so
    # the load's mask directly matches the buffer's declared bound.
    byte_idx = row // 8
    bit_idx = row % 8
    valid_len = (n + 7) // 8
    load_mask = in_range & (byte_idx < valid_len)
    byte = tl.load(valid + byte_idx, mask=load_mask, other=0)
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
