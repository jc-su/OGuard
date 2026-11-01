#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    int32_t* __restrict__ out,
    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Customer keys are 1‑based, fit in int32 for TPCH scale 10
        int32_t key = static_cast<int32_t>(keys[idx]);
        // Simple atomic add – one per matching order
        atomicAdd(&out[key], 1);
    }
}
