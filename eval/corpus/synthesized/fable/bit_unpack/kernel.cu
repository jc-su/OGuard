// One thread per code: the code starts at bit i * bits of the stream and
// spans at most two words.
extern "C" __global__ void kernel(const unsigned int* packed, unsigned int* out, long long n, long long w, long long bits) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long p = i * bits;
    long long word = p >> 5;
    int sh = (int)(p & 31);
    unsigned long long acc = (unsigned long long)packed[word] >> sh;
    if (sh + bits > 32) acc |= (unsigned long long)packed[word + 1] << (32 - sh);
    acc &= (1ull << bits) - 1ull;
    out[i] = (unsigned int)acc;
}
