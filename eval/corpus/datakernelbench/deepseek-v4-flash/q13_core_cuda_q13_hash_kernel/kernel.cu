#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    int64_t num_customers,
    const int64_t* __restrict__ o_custkey,
    int64_t n_orders,
    unsigned long long* __restrict__ order_counts
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_orders) {
        int64_t cust = o_custkey[tid];
        if (cust >= 1 && cust <= num_customers) {
            atomicAdd(&order_counts[cust], 1ULL);
        }
    }
}
