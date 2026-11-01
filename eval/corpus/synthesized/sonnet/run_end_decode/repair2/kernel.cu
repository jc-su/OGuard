// run_end_decode: decode an Arrow run-end encoded column.
// out[i] = run_values[j], where j is the smallest run index with
// run_ends[j] > i (upper_bound of i over run_ends). One thread per
// output row, each doing its own binary search over run_ends.
//
// The loop invariant `0 <= lo <= hi <= r` (so `mid` always lands inside
// `run_ends`) holds by induction on lo/hi across iterations, and the
// final `lo < r` (so `run_values[lo]` is in range) follows from the
// contract that `run_ends[r - 1] == n`. A min/max clamp applied to
// `mid` (or to the final `lo`) right before the read expresses the
// same invariant algebraically, but a compiler is free to prove that
// fact redundant (given the loop's own bracket) and fold the clamp
// away, leaving the read gated by nothing a static checker can use.
// Every read below is instead gated by an `if` whose condition is the
// exact bound comparison on the index used in that read, with the
// read itself lexically inside the branch, so the guard cannot be
// separated from the access it protects.
extern "C" __global__ void kernel(const double *__restrict__ run_values,
                                   const int *__restrict__ run_ends,
                                   double *__restrict__ out, long long r,
                                   long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= n) return;

  long long lo = 0, hi = r;
  while (lo < hi) {
    long long mid = (lo + hi) >> 1;
    bool advance = false;
    if (mid >= 0 && mid < r) {
      advance = (long long)run_ends[mid] <= i;
    }
    if (advance) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo >= 0 && lo < r) {
    out[i] = run_values[lo];
  } else {
    out[i] = 0.0;
  }
}
