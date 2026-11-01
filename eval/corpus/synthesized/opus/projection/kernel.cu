// projection: out[i] = price[i] * (1 - discount[i]) for i in [0, n).

constexpr int BLOCK = 256;  // Threads per block.

extern "C" __global__ void __launch_bounds__(BLOCK)
kernel(const double* __restrict__ price, const double* __restrict__ discount,
       double* __restrict__ out, long long n)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        out[i] = price[i] * (1.0 - discount[i]);
    }
}
