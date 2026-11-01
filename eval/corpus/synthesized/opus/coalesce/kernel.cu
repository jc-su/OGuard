// coalesce: out[i] = a[i] where row i of a is valid, and b[i] otherwise.
// valid is the Arrow validity bitmap of a, least significant bit first.
#include <cstdint>

extern "C" __global__ void kernel(const double* __restrict__ a,
                                  const uint8_t* __restrict__ valid,
                                  const double* __restrict__ b,
                                  double* __restrict__ out,
                                  int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const bool is_valid = (valid[i >> 3] >> (i & 7)) & 1;
        out[i] = is_valid ? a[i] : b[i];
    }
}
