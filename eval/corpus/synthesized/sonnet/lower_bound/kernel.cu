#include <cstdint>

extern "C" __global__ void kernel(const int64_t* keys, const int64_t* sorted, int32_t* out,
                                  int64_t n, int64_t s) {
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = start; i < n; i += stride) {
        const int64_t key = keys[i];
        int64_t lo = 0, hi = s;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (sorted[mid] < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        out[i] = (int32_t)lo;
    }
}
