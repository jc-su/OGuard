#include <cstdint>

__device__ __forceinline__ void atomic_add_i64(int64_t* addr, int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long int*>(addr),
              static_cast<unsigned long long int>(val));
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_custkey,
    int64_t n_orders,
    int64_t* __restrict__ counts,
    int64_t counts_len
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = tid; i < n_orders; i += stride) {
        int64_t key = o_custkey[i];
        if (key >= 0 && key < counts_len) {
            atomic_add_i64(counts + key, 1);
        }
    }
}
