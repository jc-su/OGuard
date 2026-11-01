#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ orderkeys,
    const double* __restrict__ quantities,
    double* __restrict__ sums,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = orderkeys[idx];
        double qty = quantities[idx];
        // Direct addressing: key is 1-based index into sums
        // Ensure key is within bounds (should be, based on TPC-H spec)
        if (key > 0) {
            atomicAdd(&sums[key], qty);
        }
    }
}
