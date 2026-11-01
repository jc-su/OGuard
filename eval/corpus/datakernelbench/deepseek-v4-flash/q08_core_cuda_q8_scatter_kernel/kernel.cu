#include <cstdint>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ o_year_ptr,
    const double* __restrict__ volume_ptr,
    const bool* __restrict__ is_brazil_ptr,
    double* __restrict__ sum_brazil_out,
    double* __restrict__ sum_total_out,
    int32_t* __restrict__ year_flags_out,
    int64_t n_elements
) {
    // Each thread reads an element and atomically adds to per-year bins
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    int32_t yr = o_year_ptr[idx];
    double vol = volume_ptr[idx];
    double brazil_vol = is_brazil_ptr[idx] ? vol : 0.0;

    // Use year offset from 1990 (covers TPC-H range)
    int bin = yr - 1990;
    atomicAdd(&sum_brazil_out[bin], brazil_vol);
    atomicAdd(&sum_total_out[bin], vol);
    if (bin >= 0 && bin < 100) {
        atomicExch(&year_flags_out[bin], 1);
    }
}
