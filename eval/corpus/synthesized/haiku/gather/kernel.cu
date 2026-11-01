#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const double* col, const int32_t* sel, int64_t m, int64_t n,
    double* out)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        out[i] = col[sel[i]];
    }
}
