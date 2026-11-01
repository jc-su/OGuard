// sorted_contains: match[i] = 1 if keys[i] is one of the sorted,
// distinct `members`, else 0. One thread per key, each doing its own
// binary search over members.
//
// The loop invariant `0 <= lo <= hi <= s` (so `mid` always lands inside
// `members`) holds by induction on lo/hi across iterations, but a static
// bound checker need not perform that induction: `mid` is clamped to
// [0, s) with two local min/max guards immediately before the read, so
// the bound follows from those two comparisons alone.
extern "C" __global__ void kernel(const long long *__restrict__ keys,
                                   const long long *__restrict__ members,
                                   signed char *__restrict__ match,
                                   long long n, long long s) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= n) return;

  long long key = keys[i];
  long long lo = 0, hi = s;
  while (lo < hi) {
    long long mid = (lo + hi) >> 1;
    // Inside the loop, s >= 1 always: hi only ever shrinks from s, and
    // lo < hi (the loop condition) forces hi >= 1.
    long long safe_mid = mid;
    if (safe_mid < 0) safe_mid = 0;
    if (safe_mid >= s) safe_mid = s - 1;
    if (members[safe_mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  bool found = false;
  if (s > 0) {
    long long safe_lo = lo;
    if (safe_lo < 0) safe_lo = 0;
    if (safe_lo >= s) safe_lo = s - 1;
    found = (lo < s) && (members[safe_lo] == key);
  }
  match[i] = found ? 1 : 0;
}
