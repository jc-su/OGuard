#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ rf_code_ptr,
    const int32_t* __restrict__ ls_code_ptr,
    const double* __restrict__ qty_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    const double* __restrict__ tax_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t shipdate_hi_days,
    double* __restrict__ out_sum_qty,
    double* __restrict__ out_sum_base_price,
    double* __restrict__ out_sum_disc_price,
    double* __restrict__ out_sum_charge,
    double* __restrict__ out_sum_disc,
    int64_t* __restrict__ out_count,
    int32_t* __restrict__ out_rf,
    int32_t* __restrict__ out_ls,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        if (sd <= shipdate_hi_days) {
            int32_t rf = rf_code_ptr[idx] + 1;  // shift by 1 to avoid key 0
            int32_t ls = ls_code_ptr[idx] + 1;
            int32_t gkey = rf * 256 + ls;
            
            double q = qty_ptr[idx];
            double p = price_ptr[idx];
            double d = disc_ptr[idx];
            double t = tax_ptr[idx];
            double dp = p * (1.0 - d);
            double ch = dp * (1.0 + t);
            
            atomicAdd(out_sum_qty + gkey, q);
            atomicAdd(out_sum_base_price + gkey, p);
            atomicAdd(out_sum_disc_price + gkey, dp);
            atomicAdd(out_sum_charge + gkey, ch);
            atomicAdd(out_sum_disc + gkey, d);
            atomicAdd((unsigned long long*)(out_count + gkey), 1ULL);
            if (atomicCAS(out_rf + gkey, 0, rf) == 0) {}
            if (atomicCAS(out_ls + gkey, 0, ls) == 0) {}
        }
    }
}
