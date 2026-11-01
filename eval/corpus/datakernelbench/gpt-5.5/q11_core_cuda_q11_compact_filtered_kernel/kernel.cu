#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ dense_values_ptr,
    const int32_t* __restrict__ present_ptr,
    int64_t dense_len,
    double threshold,
    int64_t* __restrict__ out_keys_ptr,
    double* __restrict__ out_values_ptr,
    unsigned long long* __restrict__ write_ptr
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < dense_len) {
        const double v = dense_values_ptr[idx];
        if (present_ptr[idx] != 0 && v > threshold) {
            const unsigned long long pos = atomicAdd(write_ptr, 1ULL);
            out_keys_ptr[pos] = idx;
            out_values_ptr[pos] = v;
        }
    }
}
