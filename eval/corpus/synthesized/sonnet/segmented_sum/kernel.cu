// GPU kernel for segmented_sum.
//
// sums[j] = sum(values[offsets[j]:offsets[j+1]]) for j in [0, g). One
// warp is assigned per segment (the CSR-vector pattern): the lanes of
// the warp stride over the segment's values, then combine their
// partial sums with a warp shuffle reduction. This keeps segments of
// very different sizes efficient without one thread, or one block,
// per segment.

const int WARP_SIZE = 32;
const unsigned FULL_MASK = 0xFFFFFFFFU;

extern "C" __global__ void kernel(const int *offsets, const double *values,
                                   double *sums, long long g) {
    long long warp_id = ((long long)blockIdx.x * blockDim.x + threadIdx.x) /
                         WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id >= g) return;

    int start = offsets[warp_id];
    int end = offsets[warp_id + 1];

    double total = 0.0;
    for (int idx = start + lane; idx < end; idx += WARP_SIZE) {
        total += values[idx];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(FULL_MASK, total, offset);
    }

    if (lane == 0) {
        sums[warp_id] = total;
    }
}
