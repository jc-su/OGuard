// dictionary_decode: out[i] = dictionary[codes[i]] for i in [0, n).
// Elementwise gather, one thread per row. `dictionary` is marked
// __restrict__ and read through the read-only cache path (`__ldg`) since
// it is shared read-only input across all threads.
extern "C" __global__ void kernel(const int *__restrict__ codes,
                                   const double *__restrict__ dictionary,
                                   double *__restrict__ out, long long n,
                                   long long d) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int code = codes[i];
    out[i] = __ldg(&dictionary[code]);
  }
}
