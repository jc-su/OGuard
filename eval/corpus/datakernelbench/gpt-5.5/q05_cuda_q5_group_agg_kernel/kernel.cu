#include <stdint.h>

#define MAX_GROUPS 32

extern "C" __global__ void kernel(
    const int64_t* __restrict__ nation_id_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    int64_t n_elements,
    int32_t num_nations,
    double* __restrict__ out_ptr
) {
    __shared__ double shared_accum[MAX_GROUPS];

    for (int i = threadIdx.x; i < num_nations; i += blockDim.x) {
        shared_accum[i] = 0.0;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t gid64 = nation_id_ptr[idx];
        if (gid64 >= 0 && gid64 < num_nations) {
            int gid = static_cast<int>(gid64);
            double revenue = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            atomicAdd(&shared_accum[gid], revenue);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_nations; i += blockDim.x) {
        double v = shared_accum[i];
        if (v != 0.0) {
            atomicAdd(&out_ptr[i], v);
        }
    }
}
