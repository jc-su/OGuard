#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ ps_suppkey,
    const int64_t* __restrict__ ps_availqty,
    const double* __restrict__ sum_qty,
    const int64_t* __restrict__ ps_group,
    int64_t n_ps,
    int64_t* __restrict__ qualifying_suppkeys,
    int64_t* __restrict__ n_qualifying
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_ps) {
        double sum_val = sum_qty[ps_group[idx]];
        if ((double)ps_availqty[idx] > 0.5 * sum_val && sum_val > 0.0) {
            int64_t pos = atomicAdd((unsigned long long*)n_qualifying, 1ULL);
            qualifying_suppkeys[pos] = ps_suppkey[idx];
        }
    }
}
