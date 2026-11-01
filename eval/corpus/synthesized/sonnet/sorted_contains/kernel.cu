// sorted_contains: match[i] = 1 if keys[i] is one of the sorted,
// distinct `members`, else 0. One thread per key, each doing its own
// binary search over members.
extern "C" __global__ void kernel(const long long *__restrict__ keys,
                                   const long long *__restrict__ members,
                                   signed char *__restrict__ match,
                                   long long n, long long s) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long long key = keys[i];
    long long lo = 0, hi = s;
    while (lo < hi) {
      long long mid = (lo + hi) >> 1;
      if (members[mid] < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    match[i] = (lo < s && members[lo] == key) ? 1 : 0;
  }
}
