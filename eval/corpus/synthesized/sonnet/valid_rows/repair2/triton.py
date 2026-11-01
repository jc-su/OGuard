import triton
import triton.language as tl

# The previous revision assigned one lane per row and derived that lane's
# byte index as `row // 8`, masking the load of `valid` with
# `byte_idx < valid_len` (`valid_len = (n + 7) // 8`, matching the buffer's
# declared length). That mask is implied by `row < n` -- `row < n` implies
# `row // 8 < (n + 7) // 8` by construction -- and the compiler is free to
# use exactly that implication to simplify the load's mask back down to
# `row < n` alone, which erases the one condition that ties the load to
# `valid`'s own length, leaving nothing a static checker can use.
#
# This revision instead assigns one lane per BYTE of the bitmap, so the
# byte index is the lane's own grid position, not a value derived from a
# separately-bounded row index: `byte_idx < valid_len` is the sole
# condition gating the load of `valid` at all, so it is not implied by,
# or foldable into, any other condition in the kernel. Each lane then
# decodes its own byte's up to 8 rows (an 8-iteration compile-time-unrolled
# loop) and reserves its output slots with one atomic per byte.
BLOCK_SIZE = 256


@triton.jit
def kernel(valid, rows, count, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    byte_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    valid_len = (n + 7) // 8
    byte_mask = byte_idx < valid_len
    byte = tl.load(valid + byte_idx, mask=byte_mask, other=0)
    base = byte_idx * 8

    local_count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for bit in range(8):
        row = base + bit
        wants = byte_mask & (row < n) & (((byte >> bit) & 1) != 0)
        local_count += wants.to(tl.int32)

    count_ptrs = count + tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    slot = tl.atomic_add(count_ptrs, local_count, mask=local_count > 0)
    for bit in range(8):
        row = base + bit
        wants = byte_mask & (row < n) & (((byte >> bit) & 1) != 0)
        tl.store(rows + slot, row.to(tl.int32), mask=wants)
        slot += wants.to(tl.int32)


TYPES = {
    "valid": "*u8",
    "rows": "*i32",
    "count": "*i32",
    "n": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE}
NUM_WARPS = 8


def run(valid, rows, count, n):
    valid_len = (n + 7) // 8
    if valid_len <= 0:
        return
    grid = (triton.cdiv(valid_len, BLOCK_SIZE),)
    kernel[grid](valid, rows, count, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
