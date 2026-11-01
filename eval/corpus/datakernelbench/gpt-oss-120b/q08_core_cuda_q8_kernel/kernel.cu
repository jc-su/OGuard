#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ year,
    const double*  __restrict__ volume,
    const uint8_t* __restrict__ is_brazil,
    const int32_t min_year,
    double* __restrict__ sum_total,
    double* __restrict__ sum_brazil,
    const int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int32_t y = year[idx];
    int32_t bin = y - min_year;               // 0 .. num_years-1
    double vol = volume[idx];
    atomicAdd(&sum_total[bin], vol);
    if (is_brazil[idx])
        atomicAdd(&sum_brazil[bin], vol);
}
