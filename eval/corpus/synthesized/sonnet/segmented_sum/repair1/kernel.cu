// GPU kernel for segmented_sum.
//
// sums[j] = sum(values[offsets[j]:offsets[j+1]]) for j in [0, g). One
// warp is assigned per segment (the CSR-vector pattern): the lanes of
// the warp stride over the segment's values, then combine their
// partial sums with a warp shuffle reduction. This keeps segments of
// very different sizes efficient without one thread, or one block,
// per segment.
//
// start, end, and the running index into values are widened to
// 64-bit as soon as they are read from the (32-bit) offsets array.
// offsets[warp_id] is data, not a compile-time constant, so the
// checker cannot rule out start landing close to INT_MAX; computing
// "start + lane" in 32-bit int, as the prior version did, can then
// overflow a signed int, which is undefined behavior and blocks any
// proof about the resulting address, alignment included. Doing the
// same arithmetic in long long, matching the numba.py version of this
// kernel (whose Python integers are already 64-bit), keeps every
// value into values[] and the address built from it within a range
// the checker can reason about directly.

const int WARP_SIZE = 32;
const unsigned FULL_MASK = 0xFFFFFFFFU;

extern "C" __global__ void kernel(const int *offsets, const double *values,
                                   double *sums, long long g) {
    long long warp_id = ((long long)blockIdx.x * blockDim.x + threadIdx.x) /
                         WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id >= g) return;

    long long start = (long long)offsets[warp_id];
    long long end = (long long)offsets[warp_id + 1];

    double total = 0.0;
    for (long long idx = start + lane; idx < end; idx += WARP_SIZE) {
        total += values[idx];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(FULL_MASK, total, offset);
    }

    if (lane == 0) {
        sums[warp_id] = total;
    }
}
