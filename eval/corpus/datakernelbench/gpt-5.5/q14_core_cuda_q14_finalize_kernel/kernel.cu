#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ sums_ptr,
    double* __restrict__ out_ptr
) {
    out_ptr[0] = 100.0 * sums_ptr[1] / sums_ptr[0];
}
