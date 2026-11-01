#include <cstdint>

#define SHARED_BINS 1024

// A small bin count with enough rows per block is counted in a per-block histogram in shared
// memory, which is added to counts once per block; otherwise rows add to counts directly.
extern "C" __global__ void kernel(const double* values, const double* bounds, int32_t* counts,
                                  int64_t n, int64_t b) {
    __shared__ int local[SHARED_BINS];
    const int tid = threadIdx.x;
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + tid;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t bins = b + 1;
    const bool priv = bins <= SHARED_BINS && n >= 4 * bins * (int64_t)gridDim.x;
    if (priv) {
        for (int64_t k = tid; k < bins; k += blockDim.x) local[k] = 0;
        __syncthreads();
    }
    for (int64_t i = start; i < n; i += stride) {
        const double v = values[i];
        int64_t lo = 0, hi = b;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (bounds[mid] <= v)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (priv)
            atomicAdd(&local[lo], 1);
        else
            atomicAdd((int*)&counts[lo], 1);
    }
    if (priv) {
        __syncthreads();
        for (int64_t k = tid; k < bins; k += blockDim.x) {
            const int c = local[k];
            if (c != 0) atomicAdd((int*)&counts[k], c);
        }
    }
}
