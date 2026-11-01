#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey_ptr,
    const int64_t* __restrict__ counts_ptr,
    int64_t* __restrict__ c_count_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t custkey = c_custkey_ptr[idx];
        c_count_ptr[idx] = counts_ptr[custkey];
    }
}
