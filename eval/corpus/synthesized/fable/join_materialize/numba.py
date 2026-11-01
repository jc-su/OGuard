from numba import cuda, int32, int64


def kernel(build, probe, offsets, out_build, out_probe, b, p):
    # One thread per output pair. The probe row that owns the pair is found by
    # a binary search over offsets, and its first match in build by a binary
    # search over build, so a probe row with many matches is still balanced.
    q = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    total = int64(offsets[p])
    if q < total:
        lo = int64(0)
        hi = p
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if int64(offsets[mid]) <= q:
                lo = mid
            else:
                hi = mid
        i = lo
        key = probe[i]
        flo = int64(0)
        fhi = b
        while flo < fhi:
            mid = (flo + fhi) >> 1
            if build[mid] < key:
                flo = mid + 1
            else:
                fhi = mid
        out_build[q] = int32(flo + (q - int64(offsets[i])))
        out_probe[q] = int32(i)


TYPES = (int64[:], int64[:], int32[:], int32[:], int32[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(build, probe, offsets, out_build, out_probe, b, p):
    total = out_build.shape[0]
    blocks = max(1, (total + 255) // 256)
    _jit()[blocks, 256](build, probe, offsets, out_build, out_probe, b, p)
