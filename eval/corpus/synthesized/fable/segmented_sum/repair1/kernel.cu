// segmented_sum: sums[j] = sum of values[offsets[j]:offsets[j + 1]].
//
// The kernel picks one of two paths per launch from the average segment length
// (offsets[g] / g):
//
// - short segments (average below WARP_MIN_AVG values): one thread per
//   segment with a serial loop;
// - long segments: one warp per segment, lanes read consecutive values
//   (coalesced) and combine their partial sums with shuffles, so a few very
//   long segments cannot stall single threads.
//
// Value positions in the warp path are 64-bit, so offsets[s] + lane cannot
// wrap when offsets[s] is close to INT_MAX and the loop bound p < end bounds
// every load.

#define WARP_MIN_AVG 32

extern "C" __global__ void kernel(const int* __restrict__ offsets,
                                  const double* __restrict__ values,
                                  double* __restrict__ sums,
                                  long long g) {
    const long long n = offsets[g];
    if (n >= g * WARP_MIN_AVG) {
        const unsigned lane = threadIdx.x & 31u;
        const long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
        for (long long s = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; s < g; s += nwarps) {
            const long long begin = offsets[s];
            const long long end = offsets[s + 1];
            double acc = 0.0;
            for (long long p = begin + lane; p < end; p += 32) acc += values[p];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0u) sums[s] = acc;
        }
    } else {
        const long long stride = (long long)gridDim.x * blockDim.x;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < g; i += stride) {
            const int end = offsets[i + 1];
            double acc = 0.0;
            for (int p = offsets[i]; p < end; ++p) acc += values[p];
            sums[i] = acc;
        }
    }
}
