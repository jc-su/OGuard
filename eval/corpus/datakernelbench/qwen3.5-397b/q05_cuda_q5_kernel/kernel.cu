#include <cuda_runtime.h>

#define MAX_NATIONS 32

extern "C" __global__ void kernel(
    const int64_t* __restrict__ nation_ids,
    const double* __restrict__ prices,
    const double* __restrict__ discounts,
    double* __restrict__ out,
    int64_t n_rows,
    int num_nations
) {
    __shared__ double s_rev[MAX_NATIONS];
    
    // Initialize shared memory
    if (threadIdx.x < num_nations) {
        s_rev[threadIdx.x] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_rows) {
        int64_t nid = nation_ids[idx];
        double rev = prices[idx] * (1.0 - discounts[idx]);
        // nid is guaranteed to be < num_nations
        atomicAdd(&s_rev[nid], rev);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < num_nations; i++) {
            atomicAdd(&out[i], s_rev[i]);
        }
    }
}
