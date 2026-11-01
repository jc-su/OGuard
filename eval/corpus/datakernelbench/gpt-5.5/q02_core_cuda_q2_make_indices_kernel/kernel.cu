#include <cstdint>

extern "C" __global__ void kernel(
    int64_t* __restrict__ out_ptr,
    int64_t n_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
                + static_cast<int64_t>(threadIdx.x);
    if (idx < n_elements) {
        out_ptr[idx] = idx;
    }
}
