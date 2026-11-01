extern "C" __global__ void kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    const int32_t* __restrict__ nation_key,
    const int32_t* __restrict__ year,
    double* __restrict__ out,
    int64_t n,
    int32_t num_nations,
    int32_t year_min,
    int32_t year_range
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double amt = price[idx] * (1.0 - discount[idx]) - supplycost[idx] * quantity[idx];
        int32_t nation = nation_key[idx];
        int32_t yr = year[idx];
        int32_t group = nation * year_range + (yr - year_min);
        atomicAdd(&out[group], amt);
    }
}
