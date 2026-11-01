"""merge_sorted: merge two sorted int64 columns, a before an equal b.

Merge path (Green, McColl, Bader / ModernGPU). Each block owns 1024
consecutive outputs. Every thread finds the block's slice of `a` and `b`
with two merge-path searches on the global arrays (a binary search along the
diagonal; ties are resolved so that a precedes an equal b), the block loads
the two slices into shared memory with coalesced loads, each thread finds
its own 4-output slice with a merge-path search in shared memory and merges
it sequentially into a shared staging buffer, and the block writes the
staging buffer to `out` with coalesced stores.

Revision: the helpers are inlined into the kernel, every probe of the
searches and every element read of the merge is guarded by an explicit
range test on its index against the size of the array it reads, the loops
over the block's slice are one statement per element, and the output index
is tested against m + p. The tests always pass under the task's contract
(both inputs are sorted, so the merge path stays inside them); they state
the bound of each access for the memory checker, which does not follow
calls or carry loop invariants.
"""

from numba import cuda, int64, uint64

TYPES = (int64[:], int64[:], int64[:], int64, int64)
THREADS = 256
ITEMS = 4
TILE = THREADS * ITEMS


@cuda.jit(device=True, inline="always")
def merge_path(a, a_len, b, b_off, b_len, na, nb, diag):
    """Number of elements of a[0:na] among the first `diag` outputs of merging
    a[0:na] with b[b_off:b_off + nb]; a_len and b_len are the sizes of a and b."""
    lo = diag - nb if diag > nb else 0
    hi = diag if diag < na else na
    while lo < hi:
        mid = int64(uint64(lo + hi) >> 1)  # the unsigned shift keeps mid >= 0
        jb = b_off + diag - 1 - mid
        if mid >= a_len or jb < 0 or jb >= b_len:
            break
        if a[mid] <= b[jb]:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cuda.jit(device=True, inline="always")
def load_slice(s, a, b, m, p, a0, b0, na, tile, l):
    """s[l] = element l of the block's slices, a[a0:a0 + na] then b[b0:b0 + tile - na]."""
    if l < tile:
        if l < na:
            ia = a0 + l
            if ia >= 0 and ia < m:
                s[l] = a[ia]
        else:
            ib = b0 + l - na
            if ib >= 0 and ib < p:
                s[l] = b[ib]


@cuda.jit(device=True, inline="always")
def merge_step(s, merged, na, nb, tile, i, j, k):
    """Output k of the thread's slice: s[i] (from a) or s[na + j] (from b), a first on ties."""
    if k < tile:
        ib = na + j
        has_a = i < na and i >= 0 and i < TILE
        has_b = j < nb and ib >= 0 and ib < TILE
        if has_a and (not has_b or s[i] <= s[ib]):
            merged[k] = s[i]
            i += 1
        elif has_b:
            merged[k] = s[ib]
            j += 1
    return i, j


@cuda.jit(device=True, inline="always")
def store_slice(out, merged, k0, total, l):
    g = k0 + l
    if g < total:
        out[g] = merged[l]


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

    a0 = merge_path(a, m, b, 0, p, m, p, k0)
    a1 = merge_path(a, m, b, 0, p, m, p, k1)
    b0 = k0 - a0
    na = a1 - a0
    nb = tile - na

    load_slice(s, a, b, m, p, a0, b0, na, tile, tid)
    load_slice(s, a, b, m, p, a0, b0, na, tile, tid + THREADS)
    load_slice(s, a, b, m, p, a0, b0, na, tile, tid + 2 * THREADS)
    load_slice(s, a, b, m, p, a0, b0, na, tile, tid + 3 * THREADS)
    cuda.syncthreads()

    d = tid * ITEMS
    if d < tile:
        i = merge_path(s, TILE, s, na, TILE, na, nb, d)
        j = d - i
        i, j = merge_step(s, merged, na, nb, tile, i, j, d)
        i, j = merge_step(s, merged, na, nb, tile, i, j, d + 1)
        i, j = merge_step(s, merged, na, nb, tile, i, j, d + 2)
        i, j = merge_step(s, merged, na, nb, tile, i, j, d + 3)
    cuda.syncthreads()

    store_slice(out, merged, k0, total, tid)
    store_slice(out, merged, k0, total, tid + THREADS)
    store_slice(out, merged, k0, total, tid + 2 * THREADS)
    store_slice(out, merged, k0, total, tid + 3 * THREADS)


_kernel = cuda.jit(kernel)


def run(a, b, out, m, p):
    blocks = max(1, (m + p + TILE - 1) // TILE)
    _kernel[blocks, THREADS](a, b, out, m, p)
