#include <cstdint>

#define SHARED_GROUPS 1024

// A small capacity with enough rows per block is reduced in a per-block copy in shared memory,
// which is merged into best once per block; otherwise rows update best directly.
extern "C" __global__ void kernel(const int32_t* group, const int64_t* value, int64_t* best,
                                  int64_t n, int64_t capacity) {
    __shared__ long long local[SHARED_GROUPS];
    const int tid = threadIdx.x;
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + tid;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    if (capacity <= SHARED_GROUPS && n >= 4 * capacity * (int64_t)gridDim.x) {
        for (int64_t k = tid; k < capacity; k += blockDim.x) local[k] = INT64_MIN;
        __syncthreads();
        for (int64_t i = start; i < n; i += stride)
            atomicMax(&local[group[i]], (long long)value[i]);
        __syncthreads();
        for (int64_t k = tid; k < capacity; k += blockDim.x) {
            const long long v = local[k];
            if (v != INT64_MIN) atomicMax((long long*)&best[k], v);
        }
    } else {
        for (int64_t i = start; i < n; i += stride)
            atomicMax((long long*)&best[group[i]], (long long)value[i]);
    }
}
