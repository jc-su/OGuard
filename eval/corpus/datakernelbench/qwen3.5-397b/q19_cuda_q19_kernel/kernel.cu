#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int* __restrict__ p_brand_id,
    const int* __restrict__ p_container_grp,
    const int* __restrict__ p_size,
    int64_t n,
    double* __restrict__ out_ptr
) {
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double thread_val = 0.0;
    
    if (idx < n) {
        int brand = p_brand_id[idx];
        int container = p_container_grp[idx];
        int size = p_size[idx];
        double qty = l_quantity[idx];
        double price = l_extendedprice[idx];
        double disc = l_discount[idx];
        
        bool match = false;
        
        // Condition 1: Brand#12 (12), SM (1), Qty 1-11, Size 1-5
        if (brand == 12 && container == 1) {
            if (qty >= 1.0 && qty <= 11.0 && size >= 1 && size <= 5) {
                match = true;
            }
        }
        // Condition 2: Brand#23 (23), MED (2), Qty 10-20, Size 1-10
        else if (brand == 23 && container == 2) {
            if (qty >= 10.0 && qty <= 20.0 && size >= 1 && size <= 10) {
                match = true;
            }
        }
        // Condition 3: Brand#34 (34), LG (3), Qty 20-30, Size 1-15
        else if (brand == 34 && container == 3) {
            if (qty >= 20.0 && qty <= 30.0 && size >= 1 && size <= 15) {
                match = true;
            }
        }
        
        if (match) {
            thread_val = price * (1.0 - disc);
        }
    }
    
    block_sum[threadIdx.x] = thread_val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_sum[0]);
    }
}
