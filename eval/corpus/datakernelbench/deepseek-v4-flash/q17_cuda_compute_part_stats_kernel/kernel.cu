#include <cuda_runtime.h>
#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey_ptr,
    const double* __restrict__ l_quantity_ptr,
    const int64_t* __restrict__ part_keys_ptr,
    double* __restrict__ sum_qty_ptr,
    double* __restrict__ cnt_qty_ptr,
    int64_t n_lineitems,
    int64_t n_parts
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ double block_cnt[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0.0;
    double my_cnt = 0.0;
    
    if (idx < n_lineitems) {
        int64_t pk = l_partkey_ptr[idx];
        double qty = l_quantity_ptr[idx];
        
        // Binary search for partkey
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
            my_sum = qty;
            my_cnt = 1.0;
        }
    }
    
    // Use atomicAdd per thread for scattered indices (each thread may map to different partkey)
    // This is necessary because lineitems are not sorted by partkey
    if (idx < n_lineitems) {
        int64_t pk = l_partkey_ptr[idx];
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
            atomicAdd(&sum_qty_ptr[found_idx], l_quantity_ptr[idx]);
            atomicAdd(&cnt_qty_ptr[found_idx], 1.0);
        }
    }
}
