// lag: out[i] = values[i - k] when i - k >= first[i] (the first row of row i's
// partition), and NaN otherwise.
#include <cmath>

extern "C" __global__ void kernel(const double* values,
                                  const int* first,
                                  double* out,
                                  long long n,
                                  long long k) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long long j = i - k;
        out[i] = (j >= first[i]) ? values[j] : nan("");
    }
}
