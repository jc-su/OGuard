// projection: out[i] = price[i] * (1 - discount[i]). Grid-stride loop, one row per iteration.

extern "C" __global__ void kernel(const double* __restrict__ price,
                                  const double* __restrict__ discount,
                                  double* __restrict__ out,
                                  long long n) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = price[i] * (1.0 - discount[i]);
    }
}
