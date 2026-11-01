// gather: out[i] = col[sel[i]] for i in [0, m). Grid-stride loop, one row per iteration.

extern "C" __global__ void kernel(const double* __restrict__ col,
                                  const int* __restrict__ sel,
                                  double* __restrict__ out,
                                  long long m,
                                  long long n) {
    (void)n;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
        out[i] = col[sel[i]];
    }
}
