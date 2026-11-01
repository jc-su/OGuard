#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ sums,
    double threshold,
    int64_t* __restrict__ out_keys,
    double* __restrict__ out_sums,
    unsigned long long* __restrict__ pos_out,
    int64_t n
) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        double s = sums[idx];
        if (s > threshold) {
            unsigned long long pos = atomicAdd(pos_out, 1ULL);
            out_keys[pos] = idx;
            out_sums[pos] = s;
        }
    }
}
