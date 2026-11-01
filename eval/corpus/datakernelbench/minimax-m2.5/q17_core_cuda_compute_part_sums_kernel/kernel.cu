extern "C" __global__ void kernel(
    const int64_t* __restrict__ part_idx,
    const double* __restrict__ quantity,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int64_t p = part_idx[tid];
        double q = quantity[tid];
        atomicAdd(&sum_qty[p], q);
        atomicAdd(&cnt_qty[p], 1.0);
    }
}
