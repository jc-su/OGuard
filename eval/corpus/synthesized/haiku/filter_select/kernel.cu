#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const double* quantity, int64_t n, double threshold,
    int32_t* sel, int32_t* count)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (quantity[i] < threshold) {
            int32_t idx = atomicAdd(count, 1);
            if (idx < n) {
                sel[idx] = i;
            }
        }
    }
}
