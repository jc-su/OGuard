#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    const unsigned long long* __restrict__ order_counts,
    int64_t* __restrict__ c_count,
    int64_t n_customers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_customers) {
        int64_t custkey = c_custkey[idx];
        c_count[idx] = static_cast<int64_t>(order_counts[custkey]);
    }
}
