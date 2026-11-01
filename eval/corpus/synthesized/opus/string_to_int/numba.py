"""string_to_int: parse string i as an optional sign followed by one or more ASCII
digits (any number of leading zeros) into int64; out[i] = value and valid bit i = 1
when it parses and fits, else out[i] = 0 and the bit stays 0.

The magnitude accumulates in uint64 and is checked before each step against the
limit 2^63 - 1 (or 2^63 for a minus sign): mag * 10 + d <= limit exactly when
mag <= (limit - d) // 10. Parsing stops at the first invalid byte. Each warp parses
32 consecutive rows starting at a multiple of 32 and assembles their valid bits with
a ballot; lanes 0..3 store the four bitmap bytes, so each byte has one writer and
the bits at positions n and above are 0.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32
PLUS = 43
MINUS = 45
ZERO = 48

TYPES = (
    types.int32[:],  # offsets
    types.uint8[:],  # chars
    types.int64[:],  # out
    types.uint8[:],  # valid
    types.int64,  # n
)


def kernel(offsets, chars, out, valid, n):
    lane = cuda.threadIdx.x & 31
    warp = cuda.grid(1) >> 5
    num_warps = cuda.gridsize(1) >> 5
    out_bytes = (n + 7) >> 3
    ten = types.uint64(10)
    base = warp * 32
    while base < n:
        i = base + lane
        ok = False
        if i < n:
            p = types.int64(offsets[i])
            end = types.int64(offsets[i + 1])
            neg = False
            if p < end and (chars[p] == PLUS or chars[p] == MINUS):
                neg = chars[p] == MINUS
                p += 1
            ok = p < end
            limit = types.uint64(9223372036854775807)
            if neg:
                limit = types.uint64(9223372036854775808)
            mag = types.uint64(0)
            while ok and p < end:
                c = chars[p]
                if c < ZERO or c > ZERO + 9:
                    ok = False
                else:
                    d = types.uint64(c - ZERO)
                    if mag > (limit - d) // ten:
                        ok = False
                    else:
                        mag = mag * ten + d
                p += 1
            if not ok:
                out[i] = 0
            elif neg:
                out[i] = types.int64(types.uint64(0) - mag)
            else:
                out[i] = types.int64(mag)
        word = cuda.ballot_sync(0xFFFFFFFF, ok)
        byte = (base >> 3) + lane
        if lane < 4 and byte < out_bytes:
            valid[byte] = (word >> (8 * lane)) & 0xFF
        base += num_warps * 32


_kernel = cuda.jit(kernel)


def run(offsets, chars, out, valid, n):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](offsets, chars, out, valid, n)
