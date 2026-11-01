#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* shipdate, const double* discount, const double* quantity, const double* price,
    int64_t n,
    int32_t date_lo, int32_t date_hi, double disc_lo, double disc_hi, double qty_max,
    double* result)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    while (i < n) {
        if (date_lo <= shipdate[i] && shipdate[i] < date_hi &&
            disc_lo <= discount[i] && discount[i] <= disc_hi &&
            quantity[i] < qty_max) {
            atomicAdd(result, price[i] * discount[i]);
        }
        i += stride;
    }
}
