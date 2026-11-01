// string_contains: match[i] = 1 if `pattern` occurs as a byte substring of
// string i, else 0. The empty pattern matches every string.
//
// One thread per row does a brute-force substring scan of its own string.
// `pattern` is small and shared by every thread in the block, so it is
// staged into dynamic shared memory once per block instead of every
// thread re-reading it from global memory on every candidate offset.
extern "C" __global__ void kernel(const int *__restrict__ offsets,
                                   const unsigned char *__restrict__ chars,
                                   const unsigned char *__restrict__ pattern,
                                   signed char *__restrict__ match,
                                   long long n, long long k) {
  extern __shared__ unsigned char shared_pattern[];
  for (long long j = threadIdx.x; j < k; j += blockDim.x) {
    shared_pattern[j] = pattern[j];
  }
  __syncthreads();

  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

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
      if (chars[start + s + j] != shared_pattern[j]) {
        ok = false;
        break;
      }
    }
    found = ok;
  }
  match[i] = found ? 1 : 0;
}
