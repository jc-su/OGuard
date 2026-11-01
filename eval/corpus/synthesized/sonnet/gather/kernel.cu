// GPU kernel for gather.
//
// out[i] = col[sel[i]] for i in [0, m). Every value in sel is a row id
// of col, in [0, n).

extern "C" __global__ void kernel(const double *col, const int *sel,
                                   double *out, long long m, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        out[i] = col[sel[i]];
    }
}
