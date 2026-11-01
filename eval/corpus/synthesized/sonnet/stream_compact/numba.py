from numba import cuda, int8, int32, int64

TYPES = (int64[:], int8[:], int64[:], int32[:], int64)


def kernel(values, keep, out, count, n):
    # Warp-aggregated compaction: one atomic per warp that has selected rows.
    # Every thread of a block runs the same number of loop iterations, so the
    # full-warp ballot and shuffle are always reached by all 32 lanes.
    tid = cuda.threadIdx.x
    lane = tid & 31
    stride = cuda.gridDim.x * cuda.blockDim.x
    base = cuda.blockIdx.x * cuda.blockDim.x
    while base < n:
        i = base + tid
        pred = i < n and keep[i] != 0
        mask = cuda.ballot_sync(0xFFFFFFFF, pred)
        votes = cuda.popc(mask)
        if votes > 0:
            pos = 0
            if lane == 0:
                pos = cuda.atomic.add(count, 0, votes)
            pos = cuda.shfl_sync(0xFFFFFFFF, pos, 0)
            if pred:
                rank = cuda.popc(mask & ((1 << lane) - 1))
                out[pos + rank] = values[i]
        base += stride


def run(values, keep, out, count, n):
    blocks = max(1, (n + 255) // 256)
    cuda.jit(kernel)[blocks, 256](values, keep, out, count, n)
