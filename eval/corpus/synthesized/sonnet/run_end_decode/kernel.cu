// run_end_decode: decode an Arrow run-end encoded column.
// out[i] = run_values[j], where j is the smallest run index with
// run_ends[j] > i (upper_bound of i over run_ends). One thread per
// output row, each doing its own binary search over run_ends.
extern "C" __global__ void kernel(const double *__restrict__ run_values,
                                   const int *__restrict__ run_ends,
                                   double *__restrict__ out, long long r,
                                   long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long long lo = 0, hi = r;
    while (lo < hi) {
      long long mid = (lo + hi) >> 1;
      if ((long long)run_ends[mid] <= i) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    out[i] = run_values[lo];
  }
}
