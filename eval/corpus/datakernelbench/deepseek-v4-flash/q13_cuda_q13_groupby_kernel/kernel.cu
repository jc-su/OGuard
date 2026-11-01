#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_count_ptr,
    int64_t* __restrict__ custdist_out_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t count = c_count_ptr[idx];
        atomicAdd((unsigned long long*)&custdist_out_ptr[count], 1ULL);
    }
}
