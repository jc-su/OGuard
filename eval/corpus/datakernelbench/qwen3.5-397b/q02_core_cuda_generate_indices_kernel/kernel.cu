#include <cuda_runtime.h>

extern "C" __global__ void kernel(int64_t* out_ptr, int64_t n, int64_t limit) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t k = (n < limit) ? n : limit;
    if (idx < k) {
        out_ptr[idx] = idx;
    }
}
