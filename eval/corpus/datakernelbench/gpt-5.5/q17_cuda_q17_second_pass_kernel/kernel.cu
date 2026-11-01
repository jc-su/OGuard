#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const int32_t* __restrict__ part_map,
    const double* __restrict__ sum_qty,
    const double* __restrict__ cnt_qty,
    double* __restrict__ out,
    int64_t n_lineitem,
    int64_t map_size
) {
    __shared__ double block_sum[BLOCK_SIZE];

    double local = 0.0;
    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_lineitem;
         row += (int64_t)blockDim.x * gridDim.x) {
        int64_t key = l_partkey[row];
        if (key >= 0 && key < map_size) {
            int32_t pidx = part_map[key];
            if (pidx >= 0) {
                double cnt = cnt_qty[pidx];
                if (cnt > 0.0) {
                    double qty = l_quantity[row];
                    double threshold = 0.2 * (sum_qty[pidx] / cnt);
                    if (qty < threshold) {
                        local += l_extendedprice[row];
                    }
                }
            }
        }
    }

    block_sum[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}
