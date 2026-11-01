// GPU kernel for projection.
//
// out[i] = price[i] * (1 - discount[i]) for i in [0, n).

extern "C" __global__ void kernel(const double *price, const double *discount,
                                   double *out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = price[i] * (1.0 - discount[i]);
    }
}
