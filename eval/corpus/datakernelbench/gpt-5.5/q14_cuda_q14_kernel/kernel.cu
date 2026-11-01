#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const int32_t* __restrict__ shipdate_days,
    const int64_t* __restrict__ part_keys,
    const uint8_t* __restrict__ promo_lookup,
    int64_t lookup_n,
    int64_t min_key,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ out,
    int64_t n
) {
    __shared__ double block_promo[BLOCK_SIZE];
    __shared__ double block_total[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    double my_promo = 0.0;
    double my_total = 0.0;

    if (idx < n) {
        int32_t sd = shipdate_days[idx];

        if (sd >= date_lo && sd < date_hi) {
            int64_t off = part_keys[idx] - min_key;

            if (off >= 0 && off < lookup_n) {
                uint8_t status = promo_lookup[off];

                if (status != static_cast<uint8_t>(0)) {
                    double rev = price[idx] * (1.0 - discount[idx]);
                    my_total = rev;
                    if (status == static_cast<uint8_t>(2)) {
                        my_promo = rev;
                    }
                }
            }
        }
    }

    block_promo[tid] = my_promo;
    block_total[tid] = my_total;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_promo[tid] += block_promo[tid + s];
            block_total[tid] += block_total[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, block_promo[0]);
        atomicAdd(out + 1, block_total[0]);
    }
}
