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
extern "C" __global__ void kernel(const long long *__restrict__ a,
                                   const long long *__restrict__ b,
                                   long long *__restrict__ out, long long m,
                                   long long p) {
  long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = m + p;
  if (k >= total) return;

  long long lo = max(0LL, k - p);
  long long hi = min(k, m);
  while (lo < hi) {
    long long i = lo + (hi - lo + 1) / 2;
    long long j = k - i;
    if (i > 0 && j < p && a[i - 1] > b[j]) {
      hi = i - 1;
    } else {
      lo = i;
    }
  }
  long long i = lo;
  long long j = k - i;
  if (i < m && (j >= p || a[i] <= b[j])) {
    out[k] = a[i];
  } else {
    out[k] = b[j];
  }
}
