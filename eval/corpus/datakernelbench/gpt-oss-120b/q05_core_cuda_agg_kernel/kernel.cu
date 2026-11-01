#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ nation_id,
    const double*  __restrict__ extprice,
    const double*  __restrict__ discount,
    double*        __restrict__ out,
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t nid   = nation_id[idx];
        double rev    = extprice[idx] * (1.0 - discount[idx]);
        // One atomic per row is acceptable – only ~25 distinct keys.
        atomicAdd(&out[nid], rev);
    }
}
