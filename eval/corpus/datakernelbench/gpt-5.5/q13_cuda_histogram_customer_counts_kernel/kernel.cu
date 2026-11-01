#include <cstdint>

__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long int*>(addr),
              static_cast<unsigned long long int>(val));
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    const int64_t* __restrict__ counts,
    int64_t n_customers,
    int64_t counts_len,
    int64_t* __restrict__ hist,
    int64_t hist_len
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = tid; i < n_customers; i += stride) {
        int64_t key = c_custkey[i];
        int64_t cnt = 0;
        if (key >= 0 && key < counts_len) {
            cnt = counts[key];
        }
        if (cnt >= 0 && cnt < hist_len) {
            atomic_add_i64(hist + cnt, 1);
        }
    }
}
