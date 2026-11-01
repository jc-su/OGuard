#include <cuda.h>
#include <cuda_runtime.h>

#ifndef MAX_GROUPS
#define MAX_GROUPS 64
#endif

extern "C" __global__ void kernel(
    const int64_t * __restrict__ nation_id,
    const double  * __restrict__ price,
    const double  * __restrict__ discount,
    double        * __restrict__ out,
    int64_t         n,
    int64_t         num_groups)
{
    // Shared‑memory buffer: one accumulator per group.
    __shared__ double s_out[MAX_GROUPS];

    // Initialise shared buffer.
    for (int i = threadIdx.x; i < num_groups; i += blockDim.x) {
        s_out[i] = 0.0;
    }
    __syncthreads();

    // Process assigned rows.
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t g = nation_id[idx];                 // group identifier
        double rev = price[idx] * (1.0 - discount[idx]); // revenue contribution
        atomicAdd(&s_out[g], rev);                 // lock‑free per‑group update
    }
    __syncthreads();

    // Flush block‑level results to global memory (one atomic per group per block).
    for (int i = threadIdx.x; i < num_groups; i += blockDim.x) {
        atomicAdd(&out[i], s_out[i]);
    }
}
