#include <cuda_runtime.h>
#include <cstdint>
#define MAX_GROUPS 64  // More than enough for years (max ~100 years)

extern "C" __global__ void kernel(
    const int32_t* __restrict__ o_year_ptr,
    const double* __restrict__ volume_ptr,
    const int32_t* __restrict__ is_brazil_ptr,
    double* __restrict__ sum_brazil_out,
    double* __restrict__ sum_total_out,
    int32_t* __restrict__ unique_years,
    int32_t num_groups,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double block_brazil[MAX_GROUPS];
    __shared__ double block_total[MAX_GROUPS];
    
    if (threadIdx.x < MAX_GROUPS) {
        block_brazil[threadIdx.x] = 0.0;
        block_total[threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    if (idx < n_elements) {
        int32_t year = o_year_ptr[idx];
        double vol = volume_ptr[idx];
        double brazil_vol = is_brazil_ptr[idx] ? vol : 0.0;
        
        // Find group index for this year
        for (int g = 0; g < num_groups; g++) {
            if (unique_years[g] == year) {
                atomicAdd(&block_brazil[g], brazil_vol);
                atomicAdd(&block_total[g], vol);
                break;
            }
        }
    }
    __syncthreads();
    
    // Write block results to global
    if (threadIdx.x < num_groups) {
        atomicAdd(&sum_brazil_out[threadIdx.x], block_brazil[threadIdx.x]);
        atomicAdd(&sum_total_out[threadIdx.x], block_total[threadIdx.x]);
    }
}
