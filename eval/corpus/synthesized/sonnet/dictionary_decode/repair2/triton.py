import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(codes, dictionary, out, n, d, BLOCK_SIZE: tl.constexpr):
    # Compute the flat offset in 64-bit arithmetic. program_id and arange
    # are 32-bit by default, and for an adversarial (oversized) grid
    # `pid * BLOCK_SIZE` can overflow a 32-bit accumulator; on overflow the
    # wrapped value can go negative and still pass a 32-bit `offs < n`
    # check, producing a real out-of-bounds address. Casting to int64
    # before the multiply keeps `offs` exact (and non-negative) for any
    # grid, since gridDim.x * BLOCK_SIZE fits comfortably in 64 bits.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n

    code_vals = tl.load(codes + offs, mask=mask, other=0)
    # The load from `dictionary` must be masked by a condition that
    # mentions `code_vals` itself -- the same value used in the pointer
    # arithmetic `dictionary + code_vals` -- rather than by a separately
    # clamped substitute value (a `tl.where` result) under the unrelated
    # `mask`. A checker proving this access is inside `dictionary` needs
    # the predicate guarding the load to say, directly, that the value
    # added to the pointer is within [0, d); it cannot follow a clamp
    # performed on a different value that never reaches the load.
    idx_mask = mask & (code_vals >= 0) & (code_vals < d)
    vals = tl.load(dictionary + code_vals, mask=idx_mask, other=0.0)
    tl.store(out + offs, vals, mask=mask)


TYPES = {
    "codes": "*i32",
    "dictionary": "*fp64",
    "out": "*fp64",
    "n": "i64",
    "d": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE}
NUM_WARPS = 8


def run(codes, dictionary, out, n, d):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](codes, dictionary, out, n, d, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
