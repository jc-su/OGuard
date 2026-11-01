// gather: out[i] = col[sel[i]] for i in [0, m).
//
// Every value of sel must be a row id of col, that is, in [0, n). The kernel
// does not read n; the argument gives the length of col.

#include <cstdint>

constexpr int BLOCK = 256;  // Threads per block.

extern "C" __global__ void __launch_bounds__(BLOCK)
kernel(const double* __restrict__ col, const int32_t* __restrict__ sel,
       double* __restrict__ out, long long m, long long n)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < m;
         i += stride) {
        out[i] = col[sel[i]];
    }
}
