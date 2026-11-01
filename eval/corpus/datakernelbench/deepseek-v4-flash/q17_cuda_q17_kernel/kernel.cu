#include <cuda_runtime.h>
#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey_ptr,
    const double* __restrict__ l_quantity_ptr,
    const double* __restrict__ l_extendedprice_ptr,
    const int64_t* __restrict__ part_keys_ptr,
    const double* __restrict__ avg_qty_ptr,
    const double* __restrict__ threshold_ptr,
    double* __restrict__ out_ptr,
    int64_t n_lineitems,
    int64_t n_parts
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (idx < n_lineitems) {
        int64_t pk = l_partkey_ptr[idx];
        double qty = l_quantity_ptr[idx];
        double price = l_extendedprice_ptr[idx];
        
        // Binary search for partkey in sorted part_keys
        int64_t lo = 0, hi = n_parts;
        int64_t found_idx = -1;
        
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            int64_t mid_val = part_keys_ptr[mid];
            if (mid_val == pk) {
                found_idx = mid;
                break;
            } else if (mid_val < pk) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        
        if (found_idx >= 0) {
            double avg = avg_qty_ptr[found_idx];
            double thresh = 0.2 * avg;
            if (qty < thresh) {
                my_rev = price;
            }
        }
    }
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}
