#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const int64_t* __restrict__ l_suppkey_ptr,
    const int64_t* __restrict__ sorted_supplier_ptr,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_elements,
    int64_t n_suppliers,
    double* __restrict__ out_totals
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    int32_t sd = shipdate_ptr[idx];
    if (sd >= date_lo && sd < date_hi) {
        double price_v = price_ptr[idx];
        double disc_v = discount_ptr[idx];
        double revenue = price_v * (1.0 - disc_v);
        
        // Binary search in sorted supplier keys
        int64_t target = l_suppkey_ptr[idx];
        const int64_t* lo = sorted_supplier_ptr;
        const int64_t* hi = sorted_supplier_ptr + n_suppliers;
        while (lo < hi) {
            const int64_t* mid = lo + ((hi - lo) >> 1);
            if (*mid < target) lo = mid + 1;
            else hi = mid;
        }
        
        if (lo < sorted_supplier_ptr + n_suppliers && *lo == target) {
            int64_t supplier_idx = (int64_t)(lo - sorted_supplier_ptr);
            atomicAdd(&out_totals[supplier_idx], revenue);
        }
    }
}
