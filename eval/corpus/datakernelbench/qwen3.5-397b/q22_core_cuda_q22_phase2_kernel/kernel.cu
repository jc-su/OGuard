#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_CNTRY_CODES 7

__device__ bool binary_search(const int64_t* sorted_keys, int64_t n, int64_t key) {
    int64_t lo = 0, hi = n - 1;
    while (lo <= hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (sorted_keys[mid] == key) return true;
        if (sorted_keys[mid] < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

__device__ int get_code_index(int32_t code, const int32_t* target_codes) {
    for (int i = 0; i < MAX_CNTRY_CODES; i++) {
        if (target_codes[i] == code) return i;
    }
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool* __restrict__ in_cntrycode_mask,
    const int64_t* __restrict__ o_custkey_sorted,
    const int32_t* target_codes,
    int64_t n_customers,
    int64_t n_orders,
    double avg_bal,
    int32_t* group_codes,
    int64_t* group_numcust,
    double* group_totacctbal
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_customers) {
        if (in_cntrycode_mask[idx] && c_acctbal[idx] > avg_bal) {
            if (!binary_search(o_custkey_sorted, n_orders, c_custkey[idx])) {
                int32_t code = cntrycode[idx];
                int code_idx = get_code_index(code, target_codes);
                
                if (code_idx >= 0 && code_idx < MAX_CNTRY_CODES) {
                    atomicAdd((unsigned long long int*)&group_numcust[code_idx], (unsigned long long int)1);
                    atomicAdd(&group_totacctbal[code_idx], c_acctbal[idx]);
                    group_codes[code_idx] = code;
                }
            }
        }
    }
}
