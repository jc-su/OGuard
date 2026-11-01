#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ qty,
    double threshold,
    int64_t* __restrict__ out_keys,
    double* __restrict__ out_sums,
    unsigned long long* __restrict__ pos_out,
    int64_t n
) {
    int tid = threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n; idx += stride) {
        int64_t k = keys[idx];
        bool is_start = (idx == 0) || (keys[idx - 1] != k);
        if (is_start) {
            double sum = qty[idx];
            int64_t j = idx + 1;
            while (j < n && keys[j] == k) {
                sum += qty[j];
                ++j;
            }
            if (sum > threshold) {
                unsigned long long pos = atomicAdd(pos_out, 1ULL);
                out_keys[pos] = k;
                out_sums[pos] = sum;
            }
        }
    }
}
