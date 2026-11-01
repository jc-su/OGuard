"""merge_sorted: merge two sorted int64 columns, a before an equal b.

Merge path (Green, McColl, Bader / ModernGPU). Each block owns 1024
consecutive outputs. Every thread finds the block's slice of `a` and `b`
with two merge-path searches on the global arrays (a binary search along the
diagonal; ties are resolved so that a precedes an equal b), the block loads
the two slices into shared memory with coalesced loads, each thread finds
its own 4-output slice with a merge-path search in shared memory and merges
it sequentially into a shared staging buffer, and the block writes the
staging buffer to `out` with coalesced stores.
"""

from numba import cuda, int64

TYPES = (int64[:], int64[:], int64[:], int64, int64)
THREADS = 256
ITEMS = 4
TILE = THREADS * ITEMS


@cuda.jit(device=True)
def merge_path(a, a_off, na, b, b_off, nb, diag):
    """Number of elements of a[a_off:a_off + na] among the first `diag` merged outputs."""
    lo = diag - nb if diag > nb else 0
    hi = diag if diag < na else na
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[a_off + mid] <= b[b_off + diag - 1 - mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def kernel(a, b, out, m, p):
    s = cuda.shared.array(TILE, int64)
    merged = cuda.shared.array(TILE, int64)
    tid = cuda.threadIdx.x
    total = m + p
    k0 = cuda.blockIdx.x * TILE
    if k0 >= total:
        return
    k1 = min(k0 + TILE, total)
    tile = k1 - k0

    a0 = merge_path(a, 0, m, b, 0, p, k0)
    a1 = merge_path(a, 0, m, b, 0, p, k1)
    b0 = k0 - a0
    na = a1 - a0
    nb = k1 - a1 - b0

    l = tid
    while l < tile:
        if l < na:
            s[l] = a[a0 + l]
        else:
            s[l] = b[b0 + l - na]
        l += THREADS
    cuda.syncthreads()

    d = tid * ITEMS
    if d < tile:
        d1 = min(d + ITEMS, tile)
        i = merge_path(s, 0, na, s, na, nb, d)
        j = d - i
        for k in range(d, d1):
            if j >= nb or (i < na and s[i] <= s[na + j]):
                merged[k] = s[i]
                i += 1
            else:
                merged[k] = s[na + j]
                j += 1
    cuda.syncthreads()

    l = tid
    while l < tile:
        out[k0 + l] = merged[l]
        l += THREADS


_kernel = cuda.jit(kernel)


def run(a, b, out, m, p):
    blocks = max(1, (m + p + TILE - 1) // TILE)
    _kernel[blocks, THREADS](a, b, out, m, p)
