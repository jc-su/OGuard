__device__ __forceinline__ int64_t find_order_idx(
    const int64_t* __restrict__ orderkeys,
    int64_t n_orders,
    int64_t target
) {
    int64_t lo = 0, hi = n_orders - 1;
    while (lo <= hi) {
        int64_t mid = (lo + hi) >> 1;
        if (orderkeys[mid] < target) {
            lo = mid + 1;
        } else if (orderkeys[mid] > target) {
            hi = mid - 1;
        } else {
            return mid;
        }
    }
    return lo;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkeys,
    const int8_t*  __restrict__ l_late,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int32_t* __restrict__ lo_distinct_supp,
    const int32_t* __restrict__ lo_late_distinct_supp,
    int64_t n_l1,
    int64_t n_orders,
    unsigned long long* __restrict__ out_counts
) {
    int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n_l1) {
        int64_t my_orderkey = l_orderkeys[gid];
        int64_t idx = find_order_idx(lo_orderkeys, n_orders, my_orderkey);
        
        if (idx < n_orders) {
            int64_t distinct_supp = lo_distinct_supp[idx];
            int64_t late_distinct_supp = lo_late_distinct_supp[idx];
            
            bool is_late = l_late[gid] != 0;
            bool exists_other = distinct_supp >= 2;
            bool not_exists_other_late = late_distinct_supp <= 1;
            int64_t my_supp_id = supp_ids[gid];
            
            if (is_late && exists_other && not_exists_other_late && my_supp_id >= 0) {
                atomicAdd(&out_counts[my_supp_id], 1ULL);
            }
        }
    }
}
