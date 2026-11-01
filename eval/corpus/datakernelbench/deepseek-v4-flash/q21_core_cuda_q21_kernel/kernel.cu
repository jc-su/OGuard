#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l1_orderkey_ptr,
    const int64_t* __restrict__ l1_suppkey_ptr,
    const bool* __restrict__ l1_late_ptr,
    const int64_t* __restrict__ supp_ids_ptr,
    const int64_t* __restrict__ lo_orderkeys_ptr,
    const int64_t* __restrict__ lo_distinct_supp_ptr,
    const int64_t* __restrict__ lo_late_distinct_supp_ptr,
    int64_t num_suppliers,
    int* __restrict__ out_counts_ptr,
    int64_t n_elements,
    int64_t n_lo
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        int64_t l1_orderkey = l1_orderkey_ptr[idx];
        bool l1_late = l1_late_ptr[idx];
        int64_t supp_id = supp_ids_ptr[idx];

        // Binary search for the orderkey in lo_orderkeys
        int64_t lo = 0;
        int64_t hi = n_lo - 1;
        int64_t mid;
        int64_t found_idx = -1;
        while (lo <= hi) {
            mid = lo + (hi - lo) / 2;
            int64_t val = lo_orderkeys_ptr[mid];
            if (val == l1_orderkey) {
                found_idx = mid;
                break;
            } else if (val < l1_orderkey) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        // If found and conditions match, add 1 to supplier count
        bool keep = false;
        if (found_idx >= 0 && l1_late && supp_id >= 0 && supp_id < num_suppliers) {
            int64_t distinct_supp = lo_distinct_supp_ptr[found_idx];
            int64_t late_distinct_supp = lo_late_distinct_supp_ptr[found_idx];
            bool exists_other = (distinct_supp >= 2);
            bool not_exists_other_late = (late_distinct_supp <= 1);
            keep = l1_late && exists_other && not_exists_other_late && (supp_id >= 0);
        }

        if (keep) {
            atomicAdd(&out_counts_ptr[supp_id], 1);
        }
    }
}
