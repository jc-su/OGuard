#include <cstdint>

#define BLOCK_SIZE 256
#define MAX_NATIONS 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ nation_id,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ out,
    int64_t n_elements,
    int64_t num_nations
) {
    __shared__ double group_sums[MAX_NATIONS];

    for (int64_t g = threadIdx.x; g < num_nations; g += blockDim.x) {
        group_sums[g] = 0.0;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        int64_t gid = nation_id[i];
        if (gid >= 0 && gid < num_nations) {
            double rev = price[i] * (1.0 - discount[i]);
            atomicAdd(&group_sums[gid], rev);
        }
    }

    __syncthreads();

    for (int64_t g = threadIdx.x; g < num_nations; g += blockDim.x) {
        atomicAdd(&out[g], group_sums[g]);
    }
}
