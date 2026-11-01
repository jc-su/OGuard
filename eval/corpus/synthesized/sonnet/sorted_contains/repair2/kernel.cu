// sorted_contains: match[i] = 1 if keys[i] is one of the sorted,
// distinct `members`, else 0. One thread per key, each doing its own
// binary search over members.
//
// The loop invariant `0 <= lo <= hi <= s` (so `mid` always lands inside
// `members`) holds by induction on lo/hi across iterations. A min/max
// clamp applied to `mid` right before the read expresses the same
// invariant algebraically, but a compiler is free to prove that fact
// redundant (given the loop's own bracket) and fold the clamp away,
// leaving the read gated by nothing a static checker can use. The read
// below is instead gated by an `if` whose condition is the exact bound
// comparison on `mid`, with the read itself lexically inside the
// branch, so the guard cannot be separated from the access it
// protects.
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
    bool advance = false;
    if (mid >= 0 && mid < s) {
      advance = members[mid] < key;
    }
    if (advance) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  bool found = false;
  if (lo >= 0 && lo < s) {
    found = members[lo] == key;
  }
  match[i] = found ? 1 : 0;
}
