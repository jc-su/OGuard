#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ year_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ suppkey_ptr,
    const uint8_t* __restrict__ target_supp_ptr,
    int64_t target_supp_len,
    int32_t base_year,
    int32_t n_years,
    double* __restrict__ out_target_ptr,
    double* __restrict__ out_total_ptr,
    int64_t* __restrict__ out_count_ptr,
    int64_t n_elements
) {
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n_elements; idx += stride) {
        int32_t g = year_ptr[idx] - base_year;
        if (g >= 0 && g < n_years) {
            double vol = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            int64_t sk = suppkey_ptr[idx];
            bool is_target = (sk >= 0 && sk < target_supp_len && target_supp_ptr[sk] != 0);
            atomicAdd(&out_total_ptr[g], vol);
            if (is_target) {
                atomicAdd(&out_target_ptr[g], vol);
            }
            atomicAdd(reinterpret_cast<unsigned long long*>(&out_count_ptr[g]), 1ULL);
        }
    }
}
