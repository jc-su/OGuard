// string_contains: match[i] = 1 if `pattern` occurs as a byte substring of
// string i, else 0. The empty pattern matches every string.
//
// One thread per row does a brute-force substring scan of its own string,
// reading `pattern` directly from global memory. The original version
// staged `pattern` into a dynamically sized `extern __shared__` array once
// per block, but `k` (pattern length) is unbounded by this task's contract
// ("[0, 2^31)"), so no fixed-size shared buffer can be sized to always
// hold it, and a dynamically sized shared array has no size visible to any
// analysis of the kernel body alone (its actual byte size is a launch-time
// argument), so no access into it can be proved in bounds. Reading
// `pattern` straight from global memory drops that one block-level reuse
// but keeps every access provable.
extern "C" __global__ void kernel(const int *__restrict__ offsets,
                                   const unsigned char *__restrict__ chars,
                                   const unsigned char *__restrict__ pattern,
                                   signed char *__restrict__ match,
                                   long long n, long long k) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= n) return;

  int start = offsets[i];
  int end = offsets[i + 1];
  long long length = end - start;

  if (k == 0) {
    match[i] = 1;
    return;
  }
  if (k > length) {
    match[i] = 0;
    return;
  }

  bool found = false;
  long long last_start = length - k;
  for (long long s = 0; s <= last_start && !found; ++s) {
    bool ok = true;
    for (long long j = 0; j < k; ++j) {
      if (chars[start + s + j] != pattern[j]) {
        ok = false;
        break;
      }
    }
    found = ok;
  }
  match[i] = found ? 1 : 0;
}
