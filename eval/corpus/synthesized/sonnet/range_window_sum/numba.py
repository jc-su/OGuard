from numba import cuda, int32, int64

THREADS = 256

# One block does the whole column in three passes over `out` because the kernel has
# no scratch memory for a cross-block carry. With P the inclusive prefix sums of
# `values`, lo(i) the first row with ts >= ts[i] - range and h(i) the last row with
# ts == ts[i]:
#   pass 0 (ascending tiles):  out[i] = P[i]
#   pass 1 (descending tiles): out[i] = P[i] - P[lo(i) - 1]   (sum of rows lo(i)..i)
#   pass 2 (ascending tiles):  out[i] = out[h(i)]             (rows of one ts share a frame)
# Every pass reads a whole tile before it writes the tile. Pass 1 reads only positions
# below the tile or inside it, pass 2 only positions inside or above it, so no value
# that is still needed has been overwritten. The first and last row of a tile bracket the
# search of every row of the tile. The loops use while because the argument `range`
# shadows the builtin.


def kernel(ts, values, out, n, range):
    t = int64(cuda.threadIdx.x)
    s_sum = cuda.shared.array(256, dtype=int64)
    res = cuda.local.array(8, dtype=int64)

    # pass 0: inclusive prefix sums modulo 2^64
    carry = int64(0)
    tile = int64(0)
    while tile < n:
        base = tile + t * 8
        total = int64(0)
        for k in (0, 1, 2, 3, 4, 5, 6, 7):
            if base + k < n:
                total += values[base + k]
        s_sum[t] = total
        cuda.syncthreads()
        d = 1
        while d < 256:
            prev = int64(0)
            if t >= d:
                prev = s_sum[t - d]
            cuda.syncthreads()
            s_sum[t] += prev
            cuda.syncthreads()
            d *= 2
        state = carry
        if t > 0:
            state += s_sum[t - 1]
        next_carry = carry + s_sum[255]
        cuda.syncthreads()
        for k in (0, 1, 2, 3, 4, 5, 6, 7):
            if base + k < n:
                state += values[base + k]
                out[base + k] = state
        carry = next_carry
        tile += 2048
    cuda.syncthreads()

    # pass 1: out[i] = sum of values[lo(i) .. i]
    tile = int64(-1)
    if n > 0:
        tile = ((n - 1) // 2048) * 2048
    while tile >= 0:
        end = min(tile + 2048, n)
        # smallest p in [lo, hi] with ts[p] >= key; the 256 threads probe 256 positions per round
        lo_first_lo = int64(0)
        lo_first_hi = tile
        while lo_first_hi > lo_first_lo:
            step = (lo_first_hi - lo_first_lo + 255) // 256
            q = lo_first_lo + t * step
            less = int32(0)
            if q < lo_first_hi:
                if ts[q] < ts[tile] - range:
                    less = int32(1)
            c = int64(cuda.syncthreads_count(less))
            new_lo = lo_first_lo
            if c > 0:
                new_lo = lo_first_lo + (c - 1) * step + 1
            reach = lo_first_lo + c * step
            if reach < lo_first_hi:
                lo_first_hi = reach
            lo_first_lo = new_lo
        lo_first = lo_first_lo
        # smallest p in [lo, hi] with ts[p] >= key; the 256 threads probe 256 positions per round
        lo_last_lo = lo_first
        lo_last_hi = end - 1
        while lo_last_hi > lo_last_lo:
            step = (lo_last_hi - lo_last_lo + 255) // 256
            q = lo_last_lo + t * step
            less = int32(0)
            if q < lo_last_hi:
                if ts[q] < ts[end - 1] - range:
                    less = int32(1)
            c = int64(cuda.syncthreads_count(less))
            new_lo = lo_last_lo
            if c > 0:
                new_lo = lo_last_lo + (c - 1) * step + 1
            reach = lo_last_lo + c * step
            if reach < lo_last_hi:
                lo_last_hi = reach
            lo_last_lo = new_lo
        lo_last = lo_last_lo
        for k in (0, 1, 2, 3, 4, 5, 6, 7):
            i = tile + k * 256 + t
            if i < end:
                # the same answer for a key known to lie in [lo, hi]
                r_lo = lo_first
                r_hi = lo_last
                while r_lo < r_hi:
                    mid = r_lo + (r_hi - r_lo) // 2
                    if ts[mid] < ts[i] - range:
                        r_lo = mid + 1
                    else:
                        r_hi = mid
                lo = r_lo
                below = int64(0)
                if lo > 0:
                    below = out[lo - 1]
                res[k] = out[i] - below
        cuda.syncthreads()
        for k in (0, 1, 2, 3, 4, 5, 6, 7):
            i = tile + k * 256 + t
            if i < end:
                out[i] = res[k]
        cuda.syncthreads()
        tile -= 2048

    # pass 2: every row of one ts takes the value of the last row of that ts
    tile = int64(0)
    while tile < n:
        end = min(tile + 2048, n)
        # smallest p in [lo, hi] with ts[p] >= key; the 256 threads probe 256 positions per round
        up_first_lo = tile + 1
        up_first_hi = n
        while up_first_hi > up_first_lo:
            step = (up_first_hi - up_first_lo + 255) // 256
            q = up_first_lo + t * step
            less = int32(0)
            if q < up_first_hi:
                if ts[q] < ts[tile] + 1:
                    less = int32(1)
            c = int64(cuda.syncthreads_count(less))
            new_lo = up_first_lo
            if c > 0:
                new_lo = up_first_lo + (c - 1) * step + 1
            reach = up_first_lo + c * step
            if reach < up_first_hi:
                up_first_hi = reach
            up_first_lo = new_lo
        up_first = up_first_lo
        # smallest p in [lo, hi] with ts[p] >= key; the 256 threads probe 256 positions per round
        up_last_lo = max(up_first, end)
        up_last_hi = n
        while up_last_hi > up_last_lo:
            step = (up_last_hi - up_last_lo + 255) // 256
            q = up_last_lo + t * step
            less = int32(0)
            if q < up_last_hi:
                if ts[q] < ts[end - 1] + 1:
                    less = int32(1)
            c = int64(cuda.syncthreads_count(less))
            new_lo = up_last_lo
            if c > 0:
                new_lo = up_last_lo + (c - 1) * step + 1
            reach = up_last_lo + c * step
            if reach < up_last_hi:
                up_last_hi = reach
            up_last_lo = new_lo
        up_last = up_last_lo
        for k in (0, 1, 2, 3, 4, 5, 6, 7):
            i = tile + k * 256 + t
            if i < end:
                # the same answer for a key known to lie in [lo, hi]
                r_lo = up_first
                r_hi = up_last
                while r_lo < r_hi:
                    mid = r_lo + (r_hi - r_lo) // 2
                    if ts[mid] < ts[i] + 1:
                        r_lo = mid + 1
                    else:
                        r_hi = mid
                up = r_lo
                res[k] = out[up - 1]
        cuda.syncthreads()
        for k in (0, 1, 2, 3, 4, 5, 6, 7):
            i = tile + k * 256 + t
            if i < end:
                out[i] = res[k]
        cuda.syncthreads()
        tile += 2048


TYPES = (int64[:], int64[:], int64[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(ts, values, out, n, range):
    _kernel[1, THREADS](ts, values, out, n, range)
