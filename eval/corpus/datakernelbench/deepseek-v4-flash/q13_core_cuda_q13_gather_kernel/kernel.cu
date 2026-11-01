#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    const unsigned long long* __restrict__ order_counts,
    int64_t n_customers,
    int64_t* __restrict__ c_count
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_customers) {
        int64_t cust = c_custkey[tid];
        c_count[tid] = (int64_t)order_counts[cust];
    }
}
