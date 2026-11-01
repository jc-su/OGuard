// One thread per output byte: neighbouring threads copy neighbouring bytes.
extern "C" __global__ void kernel(const int* offsets, const unsigned char* chars,
                                  unsigned char* out, long long n, long long k) {
    long long total = n * k;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
        long long i = t / k;
        long long j = t - i * k;
        int start = offsets[i];
        out[t] = (j < offsets[i + 1] - start) ? chars[start + j] : 0;
    }
}
