// run_end_decode: decode an Arrow run-end encoded column.
// out[i] = run_values[j], where j is the smallest run index with
// run_ends[j] > i (upper_bound of i over run_ends). One thread per
// output row, each doing its own binary search over run_ends.
//
// The loop invariant `0 <= lo <= hi <= r` (so `mid` always lands inside
// `run_ends`) holds by induction on lo/hi across iterations, and the
// final `lo < r` (so `run_values[lo]` is in range) follows from the
// contract that `run_ends[r - 1] == n`, but a static bound checker need
// not perform either piece of reasoning: both `mid` and the final `lo`
// are clamped with local min/max guards immediately before the read, so
// each bound follows from those comparisons alone.
extern "C" __global__ void kernel(const double *__restrict__ run_values,
                                   const int *__restrict__ run_ends,
                                   double *__restrict__ out, long long r,
                                   long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= n) return;

  long long lo = 0, hi = r;
  while (lo < hi) {
    long long mid = (lo + hi) >> 1;
    // Inside the loop, r >= 1 always: hi only ever shrinks from r, and
    // lo < hi (the loop condition) forces hi >= 1.
    long long safe_mid = mid;
    if (safe_mid < 0) safe_mid = 0;
    if (safe_mid >= r) safe_mid = r - 1;
    if ((long long)run_ends[safe_mid] <= i) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  long long safe_lo = lo;
  if (safe_lo < 0) safe_lo = 0;
  if (safe_lo >= r) safe_lo = (r > 0) ? r - 1 : 0;
  out[i] = (r > 0) ? run_values[safe_lo] : 0.0;
}
