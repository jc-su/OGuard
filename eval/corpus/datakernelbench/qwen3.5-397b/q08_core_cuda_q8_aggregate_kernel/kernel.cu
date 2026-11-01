#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ year_idx_ptr,
    const double* __restrict__ volume_ptr,
    const bool* __restrict__ is_brazil_ptr,
    double* __restrict__ sum_brazil_ptr,
    double* __restrict__ sum_total_ptr,
    int64_t n_elements,
    int64_t num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        int32_t year_idx = year_idx_ptr[idx];
        double vol = volume_ptr[idx];
        bool is_br = is_brazil_ptr[idx];
        
        if (year_idx >= 0 && year_idx < num_groups) {
            atomicAdd(&sum_total_ptr[year_idx], vol);
            if (is_br) {
                atomicAdd(&sum_brazil_ptr[year_idx], vol);
            }
        }
    }
}
