#include <cstdint>

#define BLOCK_SIZE 256
#define MAX_RF 16
#define MAX_LS 16
#define MAX_GROUPS (MAX_RF * MAX_LS)

extern "C" __global__ void kernel(
    const int32_t* __restrict__ rf_ptr,
    const int32_t* __restrict__ ls_ptr,
    const double* __restrict__ qty_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    const double* __restrict__ tax_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t shipdate_hi_days,
    double* __restrict__ g_sum_qty,
    double* __restrict__ g_sum_base_price,
    double* __restrict__ g_sum_disc_price,
    double* __restrict__ g_sum_charge,
    double* __restrict__ g_sum_disc,
    int64_t* __restrict__ g_count,
    int64_t n_elements
) {
    __shared__ double s_sum_qty[MAX_GROUPS];
    __shared__ double s_sum_base_price[MAX_GROUPS];
    __shared__ double s_sum_disc_price[MAX_GROUPS];
    __shared__ double s_sum_charge[MAX_GROUPS];
    __shared__ double s_sum_disc[MAX_GROUPS];
    __shared__ unsigned long long s_count[MAX_GROUPS];

    const int tid = threadIdx.x;

    if (tid < MAX_GROUPS) {
        s_sum_qty[tid] = 0.0;
        s_sum_base_price[tid] = 0.0;
        s_sum_disc_price[tid] = 0.0;
        s_sum_charge[tid] = 0.0;
        s_sum_disc[tid] = 0.0;
        s_count[tid] = 0ULL;
    }
    __syncthreads();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    if (idx < n_elements) {
        const int32_t sd = shipdate_ptr[idx];
        if (sd <= shipdate_hi_days) {
            const int32_t rf = rf_ptr[idx];
            const int32_t ls = ls_ptr[idx];
            if (rf >= 0 && rf < MAX_RF && ls >= 0 && ls < MAX_LS) {
                const int g = rf * MAX_LS + ls;

                const double qty = qty_ptr[idx];
                const double price = price_ptr[idx];
                const double disc = disc_ptr[idx];
                const double tax = tax_ptr[idx];
                const double disc_price = price * (1.0 - disc);
                const double charge = disc_price * (1.0 + tax);

                atomicAdd(&s_sum_qty[g], qty);
                atomicAdd(&s_sum_base_price[g], price);
                atomicAdd(&s_sum_disc_price[g], disc_price);
                atomicAdd(&s_sum_charge[g], charge);
                atomicAdd(&s_sum_disc[g], disc);
                atomicAdd(&s_count[g], 1ULL);
            }
        }
    }

    __syncthreads();

    if (tid < MAX_GROUPS) {
        const unsigned long long c = s_count[tid];
        if (c != 0ULL) {
            atomicAdd(&g_sum_qty[tid], s_sum_qty[tid]);
            atomicAdd(&g_sum_base_price[tid], s_sum_base_price[tid]);
            atomicAdd(&g_sum_disc_price[tid], s_sum_disc_price[tid]);
            atomicAdd(&g_sum_charge[tid], s_sum_charge[tid]);
            atomicAdd(&g_sum_disc[tid], s_sum_disc[tid]);
            atomicAdd(reinterpret_cast<unsigned long long*>(&g_count[tid]), c);
        }
    }
}
