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
// hold by induction across the loop and by construction of the split.
// A min/max clamp applied to an index right before the read expresses
// the same invariant as an algebraic fact about the clamped value, but
// a compiler is free to prove that fact redundant (given the loop's
// actual bracket) and fold the clamp away, leaving the read gated by
// nothing a static checker can use. Every read of `a` or `b` below is
// instead gated by an `if` whose condition is the exact bound
// comparison on the index used in that read, with the read itself
// lexically inside the branch, so the guard cannot be separated from
// the access it protects.
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
      long long bj = j;
      if (ai >= 0 && ai < m && bj >= 0 && bj < p) {
        step_down = a[ai] > b[bj];
      }
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
  if (i >= 0 && i < m) {
    if (j >= p) {
      take_a = true;
    } else if (j >= 0 && j < p) {
      take_a = a[i] <= b[j];
    }
  }

  if (take_a) {
    if (i >= 0 && i < m) {
      out[k] = a[i];
    }
  } else {
    if (j >= 0 && j < p) {
      out[k] = b[j];
    }
  }
}
