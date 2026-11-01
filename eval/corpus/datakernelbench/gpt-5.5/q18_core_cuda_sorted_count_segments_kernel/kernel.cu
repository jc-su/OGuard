#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    int64_t n,
    double threshold,
    int64_t* __restrict__ unique_counts,
    int64_t* __restrict__ big_counts,
    int32_t* __restrict__ violation
) {
    __shared__ int u_count[BLOCK_SIZE];
    __shared__ int b_count[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

    int is_unique = 0;
    int is_big = 0;

    if (idx < n) {
        int64_t key = keys[idx];
        if (idx > 0) {
            int64_t prev = keys[idx - 1];
            if (key < prev) {
                atomicExch(violation, 1);
            }
        }

        bool start = (idx == 0) || (key != keys[idx - 1]);
        if (start) {
            is_unique = 1;
            double s = 0.0;
            int64_t j = idx;
            while (j < n && keys[j] == key) {
                s += qty[j];
                ++j;
            }
            is_big = (s > threshold) ? 1 : 0;
        }
    }

    u_count[tid] = is_unique;
    b_count[tid] = is_big;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            u_count[tid] += u_count[tid + offset];
            b_count[tid] += b_count[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unique_counts[blockIdx.x] = static_cast<int64_t>(u_count[0]);
        big_counts[blockIdx.x] = static_cast<int64_t>(b_count[0]);
    }
}
