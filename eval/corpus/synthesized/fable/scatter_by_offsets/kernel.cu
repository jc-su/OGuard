// scatter_by_offsets: out[offsets[i]] = i for every i with flags[i] != 0.
//
// offsets is the exclusive prefix sum of flags, so the selected rows land in
// increasing row order. Grid-stride loop, one row per iteration.

#include <cstdint>

extern "C" __global__ void kernel(const int8_t* __restrict__ flags,
                                  const int* __restrict__ offsets,
                                  int* __restrict__ out,
                                  long long n,
                                  long long total) {
    (void)total;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (flags[i] != 0) out[offsets[i]] = (int)i;
    }
}
