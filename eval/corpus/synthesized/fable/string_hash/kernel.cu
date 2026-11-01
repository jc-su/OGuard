// One thread per string: FNV-1a over its bytes.
extern "C" __global__ void kernel(const int* offsets, const unsigned char* chars, unsigned int* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long start = offsets[i], end = offsets[i + 1];
    unsigned int h = 2166136261u;
    for (long long p = start; p < end; ++p) h = (h ^ chars[p]) * 16777619u;
    out[i] = h;
}
