#include <cuda_runtime.h>

__device__ int64_t binary_search_left(const int64_t* arr, int64_t n, int64_t val) {
    int64_t left = 0, right = n;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        if (arr[mid] < val) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

extern "C" __global__ void kernel(
    const int64_t* l1_orderkey,
    const int64_t* l1_suppkey,
    const bool* l1_late,
    const int64_t* supp_ids,
    const int64_t* lo_orderkeys,
    const int64_t* lo_distinct_supp,
    const int64_t* lo_late_distinct_supp,
    unsigned long long* counts,
    int64_t n_rows,
    int64_t n_orders,
    int64_t num_suppliers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;
    
    int64_t orderkey = l1_orderkey[idx];
    int64_t supp_id = supp_ids[idx];
    bool late = l1_late[idx];
    
    if (supp_id < 0) return;
    if (!late) return;
    
    int64_t order_idx = binary_search_left(lo_orderkeys, n_orders, orderkey);
    if (order_idx >= n_orders || lo_orderkeys[order_idx] != orderkey) return;
    
    int64_t distinct_supp = lo_distinct_supp[order_idx];
    int64_t late_distinct_supp = lo_late_distinct_supp[order_idx];
    
    if (distinct_supp < 2) return;
    if (late_distinct_supp > 1) return;
    
    atomicAdd(&counts[supp_id], 1ULL);
}
