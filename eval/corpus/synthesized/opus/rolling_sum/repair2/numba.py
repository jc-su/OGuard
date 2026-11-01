"""rolling_sum: trailing window sums of a float64 column (Numba CUDA).

The van Herk / Gil-Werman scheme, which needs O(1) additions per output and
no subtraction (so no cancellation, unlike differences of prefix sums). Cut
the column into segments of w = window rows. Every window [i - w + 1, i]
contains exactly one segment start c, so

    out[i] = suffix(i - w + 1) + prefix(i)

where prefix(i) sums the segment of i from c up to i, and suffix(j) sums the
segment of j from j to c - 1 (zero when j = c).

Each block computes the tile of TILE outputs [t0, t0 + TILE) from the span of
TILE + w - 1 inputs that its windows cover (rows outside [0, n) read as zero),
staged in shared memory. Span position x is row g0 + x with g0 = t0 - w + 1,
and the segments start where (x + 1) % w == 0. Output u needs prefix at
x = u + w - 1 (the "prefix range" [w - 1, w - 1 + TILE), rows t0 + u) and
suffix at x = u, whose segment may end past TILE (the "suffix range"
[0, suffix_end)).

Both are segmented scans, computed in three steps:
1. Each thread scans a chunk of each range sequentially (PREFIX_CHUNK and
   SUFFIX_CHUNK positions) and keeps the chunk's total and whether the chunk
   holds a segment boundary.
2. Warp 0 scans the 256 prefix chunk totals and warp 1 the 256 suffix chunk
   totals (8 per lane sequentially, then warp shuffles), giving the carry into
   each chunk.
3. An output adds the carries to its local prefix and suffix only when its
   segment crosses the chunk boundary.

The RTX A6000 (GA102) runs float64 additions at 1/64 of the float32 rate, so
the kernel keeps the additions near 3 per output (a local prefix, a local
suffix, the carries, the final sum) plus the few of step 2.

Bounds: every access is proved by the code at the access alone, without a
minimum, a maximum or a loop invariant. A column row g is read or written
only under the two comparisons 0 <= g and g < n of the very row used. The
shared arrays are rings indexed by row: row g of the span is at
xs[g & SPAN_MASK] and its local suffix sum at suffix[g & TILE_MASK], which
holds for every row of the span because the span has at most SPAN_CAP - 1
rows (the window is at most MAX_WINDOW) and the suffix sums cover TILE
consecutive rows. The per-chunk arrays are indexed by a chunk number masked
with their size - 1. The indices are formed from rows and from the thread
index, whose values the compiler does not bound, or advanced with the mask
(slot = (slot + 1) & mask), so the masks stay in the code and each shared
address is computed from its masked index (which also keeps it a multiple of
the element size) instead of a pointer that steps with a loop. A
right-to-left pass counts up and takes the position from the end of its
chunk.
"""

import numpy as np
from numba import cuda, float64, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
TILE = 1024
MAX_WINDOW = 1024
SPAN_CAP = TILE + MAX_WINDOW  # > TILE + w - 1; a power of two
SPAN_MASK = SPAN_CAP - 1
TILE_MASK = TILE - 1
CHUNK_MASK = BLOCK - 1  # one chunk of each range per thread
PREFIX_CHUNK = TILE // BLOCK  # 4: the prefix range has TILE positions
SUFFIX_CHUNK = 8  # 256 * 8 >= TILE + w - 1
LANE_CHUNKS = BLOCK // 32  # chunk totals per lane in step 2
FULL_MASK = 0xFFFFFFFF

TYPES = (float64[:], float64[:], int64, int64)


def kernel(values: np.ndarray, out: np.ndarray, n: int, window: int) -> None:
    xs = cuda.shared.array(SPAN_CAP, float64)  # inputs; the prefix range becomes prefix sums
    suffix = cuda.shared.array(TILE, float64)  # local suffix sums of the first TILE positions
    prefix_carry = cuda.shared.array(BLOCK, float64)  # chunk totals, then carries
    suffix_carry = cuda.shared.array(BLOCK, float64)
    prefix_flag = cuda.shared.array(BLOCK, int32)
    suffix_flag = cuda.shared.array(BLOCK, int32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    chunk = tid & CHUNK_MASK
    w = window
    t0 = int64(cuda.blockIdx.x) * TILE
    span = TILE + w - 1
    g0 = t0 - w + 1

    x = tid
    while x < span:
        g = g0 + (x & SPAN_MASK)  # the row of span position x
        v = 0.0
        if g >= 0 and g < n:
            v = values[g]
        xs[g & SPAN_MASK] = v
        x += BLOCK
    # The suffix range ends with the segment of position TILE - 1, or at
    # TILE - 1 when that position starts a segment (its suffix is not used).
    q = TILE % w  # (x + 1) % w at x = TILE - 1
    suffix_end = TILE - 1 if q == 0 else TILE + w - 1 - q
    cuda.syncthreads()

    # 1a. Local suffix sums of chunk [8t, 8t + 8) of the suffix range, right to left.
    suffix_run = 0.0
    suffix_boundary = int32(0)
    base = tid * SUFFIX_CHUNK
    phase = (base + SUFFIX_CHUNK) % w  # (x + 1) % w at x = base + 7
    back = 0
    while back < SUFFIX_CHUNK:
        x = base + (SUFFIX_CHUNK - 1 - back)
        if x < suffix_end:
            g = g0 + x
            v = xs[g & SPAN_MASK]
            if phase == w - 1:
                suffix_run = v
                suffix_boundary = int32(1)
            else:
                suffix_run += v
            if x < TILE:
                suffix[g & TILE_MASK] = suffix_run
        phase = phase - 1 if phase > 0 else w - 1
        back += 1
    cuda.syncthreads()  # the suffix pass reads positions that step 1b overwrites

    # 1b. Local prefix sums of chunk [4t, 4t + 4) of the prefix range: output
    #     u = 4t + j, span position u + w - 1, row t0 + u.
    prefix_run = 0.0
    prefix_boundary = int32(0)
    base = tid * PREFIX_CHUNK
    phase = base % w  # u % w; u starts a segment when it is 0
    s = (t0 + base) & SPAN_MASK  # the ring slot of row t0 + u, advanced with a mask
    j = 0
    while j < PREFIX_CHUNK:
        v = xs[s]
        if phase == 0:
            prefix_run = v
            prefix_boundary = int32(1)
        else:
            prefix_run += v
        xs[s] = prefix_run
        phase = phase + 1 if phase < w - 1 else 0
        s = (s + 1) & SPAN_MASK
        j += 1

    prefix_carry[chunk] = prefix_run
    prefix_flag[chunk] = prefix_boundary
    suffix_carry[chunk] = suffix_run
    suffix_flag[chunk] = suffix_boundary
    cuda.syncthreads()

    # 2. Carries into the chunks: exclusive segmented scans of the chunk totals.
    #    A pair (flag, value) absorbs the pairs before it unless its flag is set.
    #    A lane walks its LANE_CHUNKS chunks with a slot that it advances with
    #    the chunk mask.
    first = (lane * LANE_CHUNKS) & CHUNK_MASK
    if tid < 32:  # forward over the prefix chunks
        v = 0.0
        f = int32(0)
        c = first
        j = 0
        while j < LANE_CHUNKS:
            v = prefix_carry[c] if prefix_flag[c] != 0 else v + prefix_carry[c]
            f = int32(f | prefix_flag[c])
            c = (c + 1) & CHUNK_MASK
            j += 1
        offset = 1
        while offset < 32:
            ov = cuda.shfl_up_sync(FULL_MASK, v, offset)
            of = cuda.shfl_up_sync(FULL_MASK, f, offset)
            if lane >= offset:
                if f == 0:
                    v += ov
                f = int32(f | of)
            offset <<= 1
        carry = cuda.shfl_up_sync(FULL_MASK, v, 1)
        if lane == 0:
            carry = 0.0
        c = first
        j = 0
        while j < LANE_CHUNKS:
            chunk_total = prefix_carry[c]
            prefix_carry[c] = carry
            carry = chunk_total if prefix_flag[c] != 0 else carry + chunk_total
            c = (c + 1) & CHUNK_MASK
            j += 1
    elif tid < 64:  # backward over the suffix chunks
        last = (first + LANE_CHUNKS - 1) & CHUNK_MASK
        v = 0.0
        f = int32(0)
        c = last
        j = 0
        while j < LANE_CHUNKS:
            v = suffix_carry[c] if suffix_flag[c] != 0 else v + suffix_carry[c]
            f = int32(f | suffix_flag[c])
            c = (c - 1) & CHUNK_MASK
            j += 1
        offset = 1
        while offset < 32:
            ov = cuda.shfl_down_sync(FULL_MASK, v, offset)
            of = cuda.shfl_down_sync(FULL_MASK, f, offset)
            if lane + offset < 32:
                if f == 0:
                    v += ov
                f = int32(f | of)
            offset <<= 1
        carry = cuda.shfl_down_sync(FULL_MASK, v, 1)
        if lane == 31:
            carry = 0.0
        c = last
        j = 0
        while j < LANE_CHUNKS:
            chunk_total = suffix_carry[c]
            suffix_carry[c] = carry
            carry = chunk_total if suffix_flag[c] != 0 else carry + chunk_total
            c = (c - 1) & CHUNK_MASK
            j += 1
    cuda.syncthreads()

    # 3. out[t0 + u] = prefix(u + w - 1) + suffix(u); suffix(u) = 0 when u starts a segment.
    step = BLOCK % w
    u_mod = tid % w  # u % w
    k = 0
    while k < PREFIX_CHUNK:
        u = (tid + k * BLOCK) & TILE_MASK
        g = t0 + u  # the output row, and the row of span position u + w - 1
        # The prefix at u + w - 1 carries in when its segment started before its chunk.
        p = xs[g & SPAN_MASK]
        if u_mod > (u & (PREFIX_CHUNK - 1)):
            p += prefix_carry[(u >> 2) & CHUNK_MASK]
        # The suffix at u carries in when its segment ends past its chunk.
        phase = 0 if u_mod + 1 == w else u_mod + 1  # (u + 1) % w
        if phase != 0:
            s = suffix[(g0 + u) & TILE_MASK]
            if u + (w - 1 - phase) > (u | (SUFFIX_CHUNK - 1)):
                s += suffix_carry[(u >> 3) & CHUNK_MASK]
            p += s
        if g >= 0 and g < n:
            out[g] = p
        u_mod += step
        if u_mod >= w:
            u_mod -= w
        k += 1


_kernel = cuda.jit(kernel)


def run(values: DeviceNDArray, out: DeviceNDArray, n: int, window: int) -> None:
    if not 1 <= window <= MAX_WINDOW:
        raise ValueError(f"window must be in [1, {MAX_WINDOW}], got {window}")
    if n == 0:
        return
    grid = (n + TILE - 1) // TILE
    _kernel[grid, BLOCK](values, out, n, window)
