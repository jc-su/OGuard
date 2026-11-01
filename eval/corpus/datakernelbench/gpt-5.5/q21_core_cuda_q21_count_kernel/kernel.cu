#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l1_orderkey,
    const bool* __restrict__ l1_late,
    const int64_t* __restrict__ supp_ids,
    const int64_t* __restrict__ lo_orderkeys,
    const int64_t* __restrict__ lo_distinct_supp,
    const int64_t* __restrict__ lo_late_distinct_supp,
    int64_t* __restrict__ counts,
    int64_t n_l1,
    int64_t n_orders,
    int64_t num_suppliers
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = tid; i < n_l1; i += stride) {
        int64_t sid = supp_ids[i];

        // Reference mask starts with l1_late and supp_ids >= 0.  Check these
        // first so non-candidate rows avoid the random binary-search loads.
        if (!l1_late[i] || sid < 0 || sid >= num_suppliers) {
            continue;
        }

        int64_t key = l1_orderkey[i];

        // lower_bound equivalent to torch.searchsorted(lo_orderkeys, key).
        int64_t lo = 0;
        int64_t hi = n_orders;
        while (lo < hi) {
            int64_t mid = lo + ((hi - lo) >> 1);
            int64_t v = lo_orderkeys[mid];
            if (v < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // The reference data construction guarantees a matching aggregate row
        // for every l1 order.  The equality guard prevents an illegal load if
        // malformed input is supplied.
        if (lo >= n_orders || lo_orderkeys[lo] != key) {
            continue;
        }

        int64_t distinct_supp = lo_distinct_supp[lo];
        int64_t late_distinct_supp = lo_late_distinct_supp[lo];

        // EXISTS another supplier and NOT EXISTS another late supplier.
        if (distinct_supp >= 2 && late_distinct_supp <= 1) {
            atomicAdd(
                reinterpret_cast<unsigned long long int*>(counts + sid),
                static_cast<unsigned long long int>(1)
            );
        }
    }
}
