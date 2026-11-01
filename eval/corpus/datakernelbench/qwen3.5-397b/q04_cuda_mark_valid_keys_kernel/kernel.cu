#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey,
    const int32_t* __restrict__ l_commit,
    const int32_t* __restrict__ l_receipt,
    uint32_t* __restrict__ bitmap,
    int64_t n_lineitem
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_lineitem) {
        if (l_commit[idx] < l_receipt[idx]) {
            int64_t key = l_orderkey[idx];
            if (key > 0) atomicOr(&bitmap[key], 1u);
        }
    }
}
