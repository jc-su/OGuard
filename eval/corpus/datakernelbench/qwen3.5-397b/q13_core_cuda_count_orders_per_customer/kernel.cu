#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_custkey,
    unsigned long long* __restrict__ order_counts,
    int64_t n_orders
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t custkey = o_custkey[idx];
        atomicAdd(&order_counts[custkey], 1ULL);
    }
}
