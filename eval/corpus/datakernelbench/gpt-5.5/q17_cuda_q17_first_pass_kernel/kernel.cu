#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const int32_t* __restrict__ part_map,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n_lineitem,
    int64_t map_size
) {
    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_lineitem;
         row += (int64_t)blockDim.x * gridDim.x) {
        int64_t key = l_partkey[row];
        if (key >= 0 && key < map_size) {
            int32_t pidx = part_map[key];
            if (pidx >= 0) {
                atomicAdd(&sum_qty[pidx], l_quantity[row]);
                atomicAdd(&cnt_qty[pidx], 1.0);
            }
        }
    }
}
