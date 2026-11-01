// scatter_by_offsets: out[offsets[i]] = i for every i with flags[i] != 0.
//
// offsets is the exclusive prefix sum of flags, so the selected rows land in
// increasing row order. Grid-stride loop, one row per iteration.
//
// The position is read as an unsigned value and compared against total before
// the store, so every store stays inside out (total slots) for any offsets.
// Under the contract the comparison always holds for a flagged row, because
// its exclusive prefix sum is below the sum of all flags.

#include <cstdint>

extern "C" __global__ void kernel(const int8_t* __restrict__ flags,
                                  const int* __restrict__ offsets,
                                  int* __restrict__ out,
                                  long long n,
                                  long long total) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (flags[i] != 0) {
            const long long pos = (unsigned)offsets[i];
            if (pos < total) out[pos] = (int)i;
        }
    }
}
