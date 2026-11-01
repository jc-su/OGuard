// gather_nullable: out[i] = col[idx[i]], and NaN where idx[i] is -1
// (the unmatched rows of a left outer join).
#include <cmath>

extern "C" __global__ void kernel(const double* col,
                                  const int* idx,
                                  double* out,
                                  long long m,
                                  long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int j = idx[i];
        out[i] = (j < 0) ? nan("") : col[j];
    }
}
