#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ quantity_ptr,
    const int32_t* __restrict__ nation_ptr,
    const int32_t* __restrict__ year_ptr,
    double* __restrict__ group_sums,
    int64_t n_elements,
    int32_t min_year,
    int32_t n_groups,
    int32_t n_years
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    // Grid-stride loop for memory-bound scan
    for (int64_t i = idx; i < n_elements; i += stride) {
        int32_t nation = nation_ptr[i];
        int32_t year = year_ptr[i];
        int32_t year_offset = year - min_year;
        
        // Validate indices
        if (nation >= 0 && nation < 25 && year_offset >= 0 && year_offset < n_years) {
            int32_t group_idx = nation * n_years + year_offset;
            
            if (group_idx >= 0 && group_idx < n_groups) {
                // Compute amount = extendedprice * (1 - discount) - supplycost * quantity
                double amount = price_ptr[i] * (1.0 - discount_ptr[i]) 
                              - supplycost_ptr[i] * quantity_ptr[i];
                
                // Direct atomic add to global memory (correct, no race conditions)
                atomicAdd(&group_sums[group_idx], amount);
            }
        }
    }
}
