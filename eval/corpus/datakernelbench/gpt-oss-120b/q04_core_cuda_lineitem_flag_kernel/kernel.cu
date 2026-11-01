#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey,
    const int32_t* __restrict__ l_commit,
    const int32_t* __restrict__ l_receipt,
    int32_t* __restrict__ flags,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (l_commit[idx] < l_receipt[idx]) {
        int64_t key = l_orderkey[idx];
        // set flag[key] = 1 (atomic OR with 1)
        atomicOr(&flags[key], 1);
    }
}
