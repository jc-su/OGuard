from numba import cuda, int64

TYPES = (int64[:], int64[:], int64, int64)

DIRECT_MAX = 64


def kernel(values, out, n, w):
    # w <= 64: each thread scans its window directly (cached reads).
    # w > 64: van Herk / Gil-Werman. A block takes one w-sized segment B, builds
    # prefix maxima of segment B and suffix maxima of segment B - 1 in shared
    # memory, and out[i] = max(prefix[i], suffix[i - w + 1]).
    tid = cuda.threadIdx.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    if w <= 64:
        i = cuda.grid(1)
        while i < n:
            lo = i - w + 1
            if lo < 0:
                lo = 0
            m = values[lo]
            for j in range(lo + 1, i + 1):
                v = values[j]
                if v > m:
                    m = v
            out[i] = m
            i += stride
    else:
        neg = -9223372036854775807 - 1
        g = cuda.shared.array(1024, int64)
        h = cuda.shared.array(1025, int64)
        gl = cuda.local.array(4, int64)
        hl = cuda.local.array(4, int64)
        nseg = (n + w - 1) // w
        seg = cuda.blockIdx.x
        while seg < nseg:
            start = seg * w
            for j in range(tid, w, 256):
                if start + j < n:
                    g[j] = values[start + j]
                else:
                    g[j] = neg
                if seg > 0:
                    h[j] = values[start - w + j]
                else:
                    h[j] = neg
            if tid == 0:
                h[w] = neg
            cuda.syncthreads()
            d = 1
            while d < w:
                kk = 0
                for j in range(tid, w, 256):
                    a = g[j]
                    if j >= d and g[j - d] > a:
                        a = g[j - d]
                    b = h[j]
                    if j + d < w and h[j + d] > b:
                        b = h[j + d]
                    gl[kk] = a
                    hl[kk] = b
                    kk += 1
                cuda.syncthreads()
                kk = 0
                for j in range(tid, w, 256):
                    g[j] = gl[kk]
                    h[j] = hl[kk]
                    kk += 1
                cuda.syncthreads()
                d <<= 1
            for j in range(tid, w, 256):
                if start + j < n:
                    a = g[j]
                    b = h[j + 1]
                    if b > a:
                        a = b
                    out[start + j] = a
            cuda.syncthreads()
            seg += cuda.gridDim.x


def run(values, out, n, w):
    if w <= DIRECT_MAX:
        blocks = (n + 255) // 256
    else:
        blocks = (n + w - 1) // w
    cuda.jit(kernel)[max(1, blocks), 256](values, out, n, w)
