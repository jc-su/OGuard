#include <cstdint>

__device__ __forceinline__ int64_t atomic_inc_i64(int64_t* addr) {
    return static_cast<int64_t>(
        atomicAdd(reinterpret_cast<unsigned long long int*>(addr), 1ULL)
    );
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ hist,
    int64_t hist_len,
    int64_t* __restrict__ unique_counts,
    int64_t* __restrict__ custdist,
    int64_t* __restrict__ n_unique
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = tid; i < hist_len; i += stride) {
        int64_t v = hist[i];
        if (v > 0) {
            int64_t pos = atomic_inc_i64(n_unique);
            unique_counts[pos] = i;
            custdist[pos] = v;
        }
    }
}
