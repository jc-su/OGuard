#include <cstdint>

#define MAX_RF 16
#define MAX_LS 16
#define MAX_GROUPS (MAX_RF * MAX_LS)

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const double* __restrict__ g_sum_qty,
    const double* __restrict__ g_sum_base_price,
    const double* __restrict__ g_sum_disc_price,
    const double* __restrict__ g_sum_charge,
    const double* __restrict__ g_sum_disc,
    const int64_t* __restrict__ g_count,
    int64_t* __restrict__ out_rf,
    int64_t* __restrict__ out_ls,
    double* __restrict__ out_sum_qty,
    double* __restrict__ out_sum_base_price,
    double* __restrict__ out_sum_disc_price,
    double* __restrict__ out_sum_charge,
    double* __restrict__ out_avg_qty,
    double* __restrict__ out_avg_price,
    double* __restrict__ out_avg_disc,
    int64_t* __restrict__ out_count,
    int64_t n_groups
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_groups) {
        const int64_t gid = group_ids[i];
        const int64_t cnt = g_count[gid];
        const double cnt_d = static_cast<double>(cnt);

        out_rf[i] = gid / MAX_LS;
        out_ls[i] = gid % MAX_LS;

        const double sum_qty = g_sum_qty[gid];
        const double sum_base_price = g_sum_base_price[gid];
        const double sum_disc_price = g_sum_disc_price[gid];
        const double sum_charge = g_sum_charge[gid];
        const double sum_disc = g_sum_disc[gid];

        out_sum_qty[i] = sum_qty;
        out_sum_base_price[i] = sum_base_price;
        out_sum_disc_price[i] = sum_disc_price;
        out_sum_charge[i] = sum_charge;
        out_avg_qty[i] = sum_qty / cnt_d;
        out_avg_price[i] = sum_base_price / cnt_d;
        out_avg_disc[i] = sum_disc / cnt_d;
        out_count[i] = cnt;
    }
}
