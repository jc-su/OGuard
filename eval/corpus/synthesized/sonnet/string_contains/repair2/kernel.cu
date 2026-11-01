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
//
// Proving `chars[start + s + j]` in bounds requires chaining several
// derived quantities (`s <= last_start = length - k` and `j < k`, so
// `start + s + j < start + length = end`); a static checker need not
// perform that multi-step algebra. Both reads below are instead gated,
// immediately at the point of the read, by an `if` that compares the
// exact index used in the read against a bound already known to fit the
// buffer (`end`, itself a value read out of `offsets`, and `k`, the
// declared length of `pattern`), so each bound follows from one direct
// comparison rather than a chain of arithmetic.
extern "C" __global__ void kernel(const int *__restrict__ offsets,
                                   const unsigned char *__restrict__ chars,
                                   const unsigned char *__restrict__ pattern,
                                   signed char *__restrict__ match,
                                   long long n, long long k) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= n) return;

  long long start = offsets[i];
  long long end = offsets[i + 1];
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
      long long pos = start + s + j;
      unsigned char c = 0;
      if (pos >= 0 && pos < end) {
        c = chars[pos];
      }
      unsigned char p = 0;
      if (j >= 0 && j < k) {
        p = pattern[j];
      }
      if (c != p) {
        ok = false;
        break;
      }
    }
    found = ok;
  }
  match[i] = found ? 1 : 0;
}
