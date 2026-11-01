#include <stdint.h>
#include <limits.h>

#define BLOCK_SIZE 1024
#define EMPTY_KEY LLONG_MIN

__device__ __forceinline__ uint64_t q17_hash_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ int64_t q17_lookup_key(
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    int64_t hash_size,
    int64_t key
) {
    uint64_t mask = static_cast<uint64_t>(hash_size - 1);
    uint64_t pos = q17_hash_u64(static_cast<uint64_t>(key)) & mask;

    #pragma unroll 1
    for (int64_t probe = 0; probe < hash_size; ++probe) {
        int64_t found = hash_keys[pos];
        if (found == key) {
            return hash_vals[pos];
        }
        if (found == EMPTY_KEY) {
            return -1;
        }
        pos = (pos + 1ULL) & mask;
    }
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    const double* __restrict__ sum_qty,
    const double* __restrict__ cnt_qty,
    double* __restrict__ out,
    int64_t n_line,
    int64_t hash_size
) {
    __shared__ double block_sum[BLOCK_SIZE];

    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    double local = 0.0;

    for (int64_t row = tid; row < n_line; row += stride) {
        int64_t part_idx = q17_lookup_key(hash_keys, hash_vals, hash_size, l_partkey[row]);
        if (part_idx >= 0) {
            double cnt = cnt_qty[part_idx];
            double sum = sum_qty[part_idx];
            double qty = l_quantity[row];

            if (cnt > 0.0 && qty * cnt < 0.2 * sum) {
                local += l_extendedprice[row];
            }
        }
    }

    block_sum[threadIdx.x] = local;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}
