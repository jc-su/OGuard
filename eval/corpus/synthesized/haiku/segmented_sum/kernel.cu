#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* offsets, const double* values, int64_t g,
    double* sums)
{
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < g) {
        int32_t start = offsets[j];
        int32_t end = offsets[j + 1];
        double total = 0.0;
        for (int32_t i = start; i < end; i++) {
            total += values[i];
        }
        sums[j] = total;
    }
}
