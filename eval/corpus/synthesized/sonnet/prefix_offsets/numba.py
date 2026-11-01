from numba import cuda, int32, int64

TYPES = (int32[:], int32[:], int64)

ITEMS = 8


def kernel(lengths, offsets, n):
    # Block 0 scans the whole column in tiles of 256 * 8 elements, carrying the
    # running total between tiles. Other blocks return, so the result does not
    # depend on the grid size. No scratch space is available in the signature.
    if cuda.blockIdx.x != 0:
        return
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    wsum = cuda.shared.array(8, int32)
    local = cuda.local.array(8, int32)
    if tid == 0:
        offsets[0] = 0
    carry = 0
    base = 0
    while base < n:
        first = base + tid * 8
        s = 0
        for k in range(8):
            v = 0
            if first + k < n:
                v = lengths[first + k]
            local[k] = v
            s += v
        x = s
        d = 1
        while d < 32:
            y = cuda.shfl_up_sync(0xFFFFFFFF, x, d)
            if lane >= d:
                x += y
            d <<= 1
        if lane == 31:
            wsum[warp] = x
        cuda.syncthreads()
        wpre = 0
        total = 0
        for w in range(8):
            if w < warp:
                wpre += wsum[w]
            total += wsum[w]
        run_sum = carry + wpre + x - s
        for k in range(8):
            run_sum += local[k]
            if first + k < n:
                offsets[first + k + 1] = run_sum
        carry += total
        cuda.syncthreads()
        base += 2048


def run(lengths, offsets, n):
    cuda.jit(kernel)[1, 256](lengths, offsets, n)
