#include <cstdint>

__device__ inline uint64_t hash_single(uint64_t s) {
    uint64_t h = s;
    h ^= h >> 32; h *= 0x85EBCA6B51CE4E43ULL;
    h ^= h >> 33; h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 33;
    return h;
}

extern "C" __global__ void kernel(
    const double* __restrict__ ps_sum_qty,
    const int64_t* __restrict__ ps_availqty,
    const int64_t* __restrict__ ps_suppkey,
    unsigned long long int* __restrict__ supp_hash_set,
    int64_t H_s, int64_t N_ps) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_ps) {
        double sum_qty = ps_sum_qty[i];
        if (sum_qty > 0.0 && (double)ps_availqty[i] > 0.5 * sum_qty) {
            uint64_t s = ps_suppkey[i];
            uint64_t h = hash_single(s) & (H_s - 1);
            while (true) {
                unsigned long long int old = atomicCAS(supp_hash_set + h, 
                                                       (unsigned long long int)-1, 
                                                       (unsigned long long int)s);
                if (old == (unsigned long long int)-1 || old == (unsigned long long int)s) break;
                h = (h + 1) & (H_s - 1);
            }
        }
    }
}
