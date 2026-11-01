// coalesce: out[i] = a[i] when row i of a is valid, else b[i].
// valid is the Arrow validity bitmap of a: row i is valid when bit i % 8 of
// byte i / 8 is 1, least significant bit first.
extern "C" __global__ void kernel(const double* a,
                                  const unsigned char* valid,
                                  const double* b,
                                  double* out,
                                  long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = ((valid[i >> 3] >> (i & 7)) & 1) ? a[i] : b[i];
    }
}
