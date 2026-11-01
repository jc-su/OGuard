#include <cstdint>

// Counts per block are kept in shared memory when the b + 1 buckets fit in
// SHARED_BINS int32 slots (16 KiB); otherwise every row updates counts
// directly. Launch with 256 threads per block and any grid size of at least
// one block.
constexpr int SHARED_BINS = 4096;

extern "C" __global__ void __launch_bounds__(256)
kernel(const double* __restrict__ values, const double* __restrict__ bounds,
       int32_t* counts, int64_t n, int64_t b)
{
    __shared__ int32_t local[SHARED_BINS];
    const int64_t buckets = b + 1;
    const bool in_shared = buckets <= SHARED_BINS;
    if (in_shared) {
        for (int64_t k = threadIdx.x; k < buckets; k += blockDim.x) {
            local[k] = 0;
        }
        __syncthreads();
    }
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const double v = values[i];
        int64_t lo = 0;
        int64_t hi = b;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (bounds[mid] <= v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (in_shared) {
            atomicAdd(&local[lo], 1);
        } else {
            atomicAdd(&counts[lo], 1);
        }
    }
    if (in_shared) {
        __syncthreads();
        for (int64_t k = threadIdx.x; k < buckets; k += blockDim.x) {
            const int32_t c = local[k];
            if (c != 0) {
                atomicAdd(&counts[k], c);
            }
        }
    }
}
