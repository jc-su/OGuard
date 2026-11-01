#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ suppkeys,
    const int64_t* __restrict__ availqty,
    const double* __restrict__ sum_qty,
    int64_t* __restrict__ output,
    int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t grp = group_ids[idx];
        double sum = sum_qty[grp];
        if (sum > 0.0) {
            double half_sum = 0.5 * sum;
            double avail = static_cast<double>(availqty[idx]);
            if (avail > half_sum) {
                output[idx] = suppkeys[idx];
                return;
            }
        }
        output[idx] = -1;
    }
}
