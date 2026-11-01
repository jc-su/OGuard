#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ out_rev,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t key = custkey[idx];
    double rev = price[idx] * (1.0 - discount[idx]);

    // atomicAdd on double is supported on sm_70+ (H100 is sm_90)
    atomicAdd(&out_rev[key], rev);
}
