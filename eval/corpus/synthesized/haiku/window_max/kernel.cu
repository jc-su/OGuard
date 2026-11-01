#include <cstdint>
#include <algorithm>

extern "C" __global__ void kernel(
    const int64_t* values,
    int64_t* out,
    int64_t n,
    int64_t w
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t start = (idx < w - 1) ? 0 : (idx - w + 1);
        int64_t end = idx + 1;
        int64_t max_val = values[start];
        for (int64_t i = start + 1; i < end; i++) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }
        out[idx] = max_val;
    }
}
