// bucketize: out[i] is the number of bounds (sorted ascending) that are less
// than or equal to values[i], found by binary search. values has no NaN.
extern "C" __global__ void kernel(const double* values,
                                  const double* bounds,
                                  int* out,
                                  long long n,
                                  long long b) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double v = values[i];
        long long lo = 0;
        long long hi = b;
        while (lo < hi) {
            long long mid = (lo + hi) >> 1;
            if (bounds[mid] <= v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[i] = (int)lo;
    }
}
