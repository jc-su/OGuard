#include <cuda_runtime.h>

#define NUM_COUNTRY_CODES 7
__device__ __constant__ int32_t d_target_codes[NUM_COUNTRY_CODES] = {13, 31, 23, 29, 30, 18, 17};

__device__ bool binary_search_exists(const int64_t* arr, int64_t n, int64_t key) {
    int64_t left = 0;
    int64_t right = n - 1;
    while (left <= right) {
        int64_t mid = left + (right - left) / 2;
        if (arr[mid] == key) return true;
        if (arr[mid] < key) left = mid + 1;
        else right = mid - 1;
    }
    return false;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool* __restrict__ in_mask,
    const int64_t* __restrict__ orders_sorted,
    int64_t n_orders,
    double avg_bal,
    int64_t* __restrict__ count_per_country,
    double* __restrict__ bal_per_country,
    int64_t n_customers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_customers) return;
    
    if (!in_mask[idx]) return;
    
    int32_t code = cntrycode[idx];
    double bal = c_acctbal[idx];
    
    if (bal <= avg_bal) return;
    
    int64_t custkey = c_custkey[idx];
    
    bool has_order = binary_search_exists(orders_sorted, n_orders, custkey);
    
    if (has_order) return;
    
    int code_idx = -1;
    #pragma unroll
    for (int i = 0; i < NUM_COUNTRY_CODES; i++) {
        if (code == d_target_codes[i]) {
            code_idx = i;
            break;
        }
    }
    
    if (code_idx >= 0) {
        atomicAdd((unsigned long long*)&count_per_country[code_idx], 1ULL);
        atomicAdd(&bal_per_country[code_idx], bal);
    }
}
