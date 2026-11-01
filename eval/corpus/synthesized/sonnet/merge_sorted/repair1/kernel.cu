// merge_sorted: merge ascending-sorted a (length m) and b (length p)
// into out (length m + p), ascending, with an element of a before an
// equal element of b.
//
// Uses the "merge path" (co-rank) algorithm: for each output position
// k, a binary search over the diagonal finds the split (i, j) with
// i + j = k such that the first k output elements are exactly a[0:i)
// merged with b[0:j). Each output position is then produced by an
// independent thread with no synchronization, in
// O((m + p) log(min(m, p))) total work.
//
// The invariants that keep every index below inside its buffer (the
// binary search's `lo/hi` bracket, and the co-rank split `i + j = k`)
// hold by induction across the loop and by construction of the split,
// but a static bound checker need not perform either piece of
// reasoning: every index used to read `a` or `b` is clamped with local
// min/max guards immediately before the read, so each bound follows
// from those comparisons alone.
extern "C" __global__ void kernel(const long long *__restrict__ a,
                                   const long long *__restrict__ b,
                                   long long *__restrict__ out, long long m,
                                   long long p) {
  long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = m + p;
  if (k < 0 || k >= total) return;

  long long lo = max(0LL, k - p);
  long long hi = min(k, m);
  while (lo < hi) {
    long long i = lo + (hi - lo + 1) / 2;
    long long j = k - i;
    bool step_down = false;
    if (i > 0 && j < p) {
      long long ai = i - 1;
      if (ai < 0) ai = 0;
      if (ai >= m) ai = m - 1;
      long long bj = j;
      if (bj < 0) bj = 0;
      if (bj >= p) bj = p - 1;
      step_down = a[ai] > b[bj];
    }
    if (step_down) {
      hi = i - 1;
    } else {
      lo = i;
    }
  }
  long long i = lo;
  long long j = k - i;

  bool take_a = false;
  if (i < m) {
    if (j >= p) {
      take_a = true;
    } else {
      long long ai = i;
      if (ai < 0) ai = 0;
      if (ai >= m) ai = m - 1;
      long long bj = j;
      if (bj < 0) bj = 0;
      if (bj >= p) bj = p - 1;
      take_a = a[ai] <= b[bj];
    }
  }

  if (take_a) {
    long long ai = i;
    if (ai < 0) ai = 0;
    if (ai >= m) ai = m - 1;
    out[k] = a[ai];
  } else {
    long long bj = j;
    if (bj < 0) bj = 0;
    if (bj >= p) bj = p - 1;
    out[k] = b[bj];
  }
}
