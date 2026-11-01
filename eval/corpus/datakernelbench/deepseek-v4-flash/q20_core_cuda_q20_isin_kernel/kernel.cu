#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ s_suppkey,
    const int64_t* __restrict__ s_row_idx,
    const int64_t* __restrict__ qual_suppkeys,
    int64_t n_s,
    int64_t n_qual,
    int64_t* __restrict__ out_indices,
    int64_t* __restrict__ n_out
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_s) {
        int64_t sk = s_suppkey[idx];
        int64_t found = 0;
        for (int64_t i = 0; i < n_qual; i++) {
            if (qual_suppkeys[i] == sk) {
                found = 1;
                break;
            }
        }
        if (found) {
            int64_t pos = atomicAdd((unsigned long long*)n_out, 1ULL);
            out_indices[pos] = s_row_idx[idx];
        }
    }
}
