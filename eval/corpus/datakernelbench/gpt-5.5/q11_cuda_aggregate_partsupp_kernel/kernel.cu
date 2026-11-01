#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    const double* __restrict__ ps_supplycost,
    const int64_t* __restrict__ ps_availqty,
    const uint8_t* __restrict__ supplier_keep,
    int64_t supplier_keep_len,
    double* __restrict__ part_sums,
    int64_t part_sums_len,
    double* __restrict__ total_sum,
    int64_t n_partsupp
) {
    __shared__ double block_total[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double local = 0.0;

    if (idx < n_partsupp) {
        int64_t sk = ps_suppkey[idx];
        if (sk >= 0 && sk < supplier_keep_len && supplier_keep[sk] != 0) {
            int64_t pk = ps_partkey[idx];
            if (pk >= 0 && pk < part_sums_len) {
                double v = ps_supplycost[idx] * static_cast<double>(ps_availqty[idx]);
                atomicAdd(&part_sums[pk], v);
                local = v;
            }
        }
    }

    block_total[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_total[threadIdx.x] += block_total[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(total_sum, block_total[0]);
    }
}
