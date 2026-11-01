// Pass 1: Filter lineitems by checking partkey against the dense O(1) part_map.
// For matched rows, accumulate sum and count of quantity per partkey.
extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const int32_t* __restrict__ part_map,
    int64_t max_partkey,
    double* __restrict__ sum_qty,
    int32_t* __restrict__ cnt_qty,
    int64_t n_lineitems
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_lineitems) return;

    int64_t pk = l_partkey[i];
    if (pk >= 0 && pk <= max_partkey) {
        int32_t p_idx = part_map[pk];
        if (p_idx != -1) {
            double qty = l_quantity[i];
            atomicAdd(&sum_qty[p_idx], qty);
            atomicAdd(&cnt_qty[p_idx], 1);
        }
    }
}
