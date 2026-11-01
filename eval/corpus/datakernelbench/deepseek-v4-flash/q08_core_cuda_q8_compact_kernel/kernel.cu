#include <cstdint>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    double* __restrict__ sum_brazil_arr,
    double* __restrict__ sum_total_arr,
    int32_t* __restrict__ year_flags,
    int32_t* __restrict__ out_years,
    double* __restrict__ out_mkt_share,
    int32_t* __restrict__ num_years_dev,
    int max_bins
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= max_bins) return;

    if (year_flags[tid] > 0) {
        double total = sum_total_arr[tid];
        if (total > 0.0) {
            int pos = atomicAdd(num_years_dev, 1);
            out_years[pos] = 1990 + tid;
            out_mkt_share[pos] = sum_brazil_arr[tid] / total;
        }
    }
}
