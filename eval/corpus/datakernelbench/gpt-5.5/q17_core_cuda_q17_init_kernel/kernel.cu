#include <stdint.h>
#include <limits.h>

#define EMPTY_KEY LLONG_MIN

extern "C" __global__ void kernel(
    int64_t* __restrict__ hash_keys,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    double* __restrict__ out,
    int64_t hash_size,
    int64_t n_parts,
    int64_t n_init
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = tid; i < n_init; i += stride) {
        if (i < hash_size) {
            hash_keys[i] = EMPTY_KEY;
        }
        if (i < n_parts) {
            sum_qty[i] = 0.0;
            cnt_qty[i] = 0.0;
        }
        if (i == 0) {
            *out = 0.0;
        }
    }
}
