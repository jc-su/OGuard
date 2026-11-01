#include <cstdint>

#define BLOCK_SIZE 512
#define MAX_GROUPS 64

extern "C" __global__ void kernel(
    const int32_t* __restrict__ returnflag,
    const int32_t* __restrict__ linestatus,
    const double* __restrict__ quantity,
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    const double* __restrict__ tax,
    const int32_t* __restrict__ shipdate_days,
    int32_t shipdate_hi_days,
    int32_t n_ls,
    int32_t n_groups,
    int64_t n,
    double* __restrict__ out_qty,
    double* __restrict__ out_price,
    double* __restrict__ out_disc_price,
    double* __restrict__ out_charge,
    double* __restrict__ out_disc,
    int64_t* __restrict__ out_count
) {
    __shared__ double sh_qty[MAX_GROUPS];
    __shared__ double sh_price[MAX_GROUPS];
    __shared__ double sh_disc_price[MAX_GROUPS];
    __shared__ double sh_charge[MAX_GROUPS];
    __shared__ double sh_disc[MAX_GROUPS];
    __shared__ unsigned long long sh_count[MAX_GROUPS];

    for (int g = threadIdx.x; g < n_groups; g += BLOCK_SIZE) {
        sh_qty[g] = 0.0;
        sh_price[g] = 0.0;
        sh_disc_price[g] = 0.0;
        sh_charge[g] = 0.0;
        sh_disc[g] = 0.0;
        sh_count[g] = 0ULL;
    }
    __syncthreads();

    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK_SIZE;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         idx < n;
         idx += stride) {
        int32_t sd = shipdate_days[idx];
        if (sd <= shipdate_hi_days) {
            int32_t rf = returnflag[idx];
            int32_t ls = linestatus[idx];
            int32_t gid = rf * n_ls + ls;
            if (rf >= 0 && ls >= 0 && gid >= 0 && gid < n_groups) {
                double qty = quantity[idx];
                double price = extendedprice[idx];
                double disc = discount[idx];
                double tax_v = tax[idx];
                double disc_price = price * (1.0 - disc);
                atomicAdd(&sh_qty[gid], qty);
                atomicAdd(&sh_price[gid], price);
                atomicAdd(&sh_disc_price[gid], disc_price);
                atomicAdd(&sh_charge[gid], disc_price * (1.0 + tax_v));
                atomicAdd(&sh_disc[gid], disc);
                atomicAdd(&sh_count[gid], 1ULL);
            }
        }
    }
    __syncthreads();

    for (int g = threadIdx.x; g < n_groups; g += BLOCK_SIZE) {
        if (sh_count[g] != 0ULL) {
            atomicAdd(out_qty + g, sh_qty[g]);
            atomicAdd(out_price + g, sh_price[g]);
            atomicAdd(out_disc_price + g, sh_disc_price[g]);
            atomicAdd(out_charge + g, sh_charge[g]);
            atomicAdd(out_disc + g, sh_disc[g]);
            atomicAdd(reinterpret_cast<unsigned long long*>(out_count + g), sh_count[g]);
        }
    }
}
