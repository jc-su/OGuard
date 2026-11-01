#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    double* __restrict__ revenue_accum,
    int* __restrict__ present_accum,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_sup,
    int64_t total_rows
) {
    __shared__ double smem[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double total = 0.0;
    int64_t current_key = -1;

    for (int64_t i = idx; i < total_rows; i += blockDim.x * gridDim.x) {
        if (l_shipdate_days[i] >= date_lo && l_shipdate_days[i] < date_hi) {
            double rev = l_extendedprice[i] * (1.0 - l_discount[i]);
            int64_t sk = l_suppkey[i];
            if (sk == current_key) {
                total += rev;
            } else {
                if (current_key != -1) {
                    atomicAdd(revenue_accum + current_key, total);
                }
                current_key = sk;
                total = rev;
            }
            atomicMax(present_accum + sk, 1);
        }
    }

    if (current_key != -1) {
        atomicAdd(revenue_accum + current_key, total);
    }
}
