from numba import cuda, int32, int64


def kernel(group, value, best, n, g):
    # One thread per row: a compare-and-swap loop on best[group] that keeps
    # the row with the larger value, or the smaller row on a tie. The loop
    # ends as soon as the published row is at least as good as this one.
    i = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    if i < n:
        k = int64(group[i])
        v = value[i]
        cur = int64(best[k])
        while True:
            if cur >= 0:
                cv = value[cur]
                if cv > v or (cv == v and cur < i):
                    break
            old = int64(cuda.atomic.cas(best, k, int32(cur), int32(i)))
            if old == cur:
                break
            cur = old


TYPES = (int32[:], int64[:], int32[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(group, value, best, n, g):
    blocks = max(1, (n + 255) // 256)
    _jit()[blocks, 256](group, value, best, n, g)
