import numpy as np
from numba import cuda
from numba.types import int32, int64


# One block walks the column tile by tile. A thread owns 8 consecutive rows per
# tile. The state of a row is the pair (partition started, running max),
# combined by (f1, v1) . (f2, v2) = (f1 | f2, f2 ? v2 : max(v1, v2)), which is
# associative, so a scan of the per-thread aggregates, with a carry over the
# tiles, gives every thread its starting state. Work is linear for any
# partition length.
def kernel(values, first, out, n):
    scan_f = cuda.shared.array(256, int32)
    scan_v = cuda.shared.array(256, int64)
    t = cuda.threadIdx.x
    carry_f = np.int32(0)
    carry_v = np.int64(-9223372036854775808)
    for tile in range(0, n, 2048):
        base = tile + t * 8
        f = np.int32(0)
        v = np.int64(-9223372036854775808)
        for k in range(8):
            i = base + k
            if i < n:
                xf = np.int32(first[i] == i)
                x = values[i]
                if xf != 0:
                    v = x
                elif x > v:
                    v = x
                f |= xf
        scan_f[t] = f
        scan_v[t] = v
        cuda.syncthreads()
        d = 1
        while d < 256:
            of = np.int32(0)
            ov = np.int64(-9223372036854775808)
            if t >= d:
                of = scan_f[t - d]
                ov = scan_v[t - d]
            cuda.syncthreads()
            if scan_f[t] != 0:
                ov = scan_v[t]
            elif scan_v[t] > ov:
                ov = scan_v[t]
            of |= scan_f[t]
            scan_f[t] = of
            scan_v[t] = ov
            cuda.syncthreads()
            d *= 2
        # State before this thread's slice: carry . inclusive state of thread t - 1.
        sf = carry_f
        sv = carry_v
        if t > 0:
            if scan_f[t - 1] != 0:
                sv = scan_v[t - 1]
            elif scan_v[t - 1] > sv:
                sv = scan_v[t - 1]
            sf |= scan_f[t - 1]
        tile_f = scan_f[255]
        tile_v = scan_v[255]
        for k in range(8):
            i = base + k
            if i < n:
                xf = np.int32(first[i] == i)
                x = values[i]
                if xf != 0:
                    sv = x
                elif x > sv:
                    sv = x
                sf |= xf
                out[i] = sv
        if tile_f != 0:
            carry_v = tile_v
        elif tile_v > carry_v:
            carry_v = tile_v
        carry_f |= tile_f
        cuda.syncthreads()


TYPES = (int64[:], int32[:], int64[:], int64)

_kernel = cuda.jit(kernel)


def run(values, first, out, n):
    _kernel[1, 256](values, first, out, n)
