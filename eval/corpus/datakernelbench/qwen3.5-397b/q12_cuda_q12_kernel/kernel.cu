#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode_ptr,
    const int32_t* __restrict__ priority_ptr,
    const int32_t* __restrict__ commit_ptr,
    const int32_t* __restrict__ receipt_ptr,
    const int32_t* __restrict__ ship_ptr,
    int32_t receipt_lo,
    int32_t receipt_hi,
    unsigned long long* __restrict__ counts_ptr,
    int64_t n
) {
    extern __shared__ unsigned long long s_counts[];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) s_counts[i] = 0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long l_counts[4] = {0, 0, 0, 0};

    if (idx < n) {
        int32_t commit = commit_ptr[idx];
        int32_t receipt = receipt_ptr[idx];
        int32_t ship = ship_ptr[idx];
        
        bool date_ok = (commit < receipt) && (ship < commit) && 
                       (receipt >= receipt_lo) && (receipt < receipt_hi);
        
        if (date_ok) {
            int32_t mode = shipmode_ptr[idx];
            int32_t prio = priority_ptr[idx];
            int bin = mode * 2 + prio;
            l_counts[bin]++;
        }
    }

    for (int i = 0; i < 4; ++i) {
        if (l_counts[i] > 0) {
            atomicAdd(&s_counts[i], l_counts[i]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            if (s_counts[i] > 0) {
                atomicAdd(&counts_ptr[i], s_counts[i]);
            }
        }
    }
}
