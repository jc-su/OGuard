#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_custkey_ptr,
    int64_t* __restrict__ counts_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t custkey = o_custkey_ptr[idx];
        atomicAdd((unsigned long long*)&counts_ptr[custkey], 1ULL);
    }
}
