// bloom_probe: match[i] = 1 when both hashed bits of keys[i] are set in the
// Bloom filter bits (32 * words bits, written by bloom_build), else 0.
extern "C" __global__ void kernel(const long long* keys,
                                  const unsigned* bits,
                                  signed char* match,
                                  long long n,
                                  long long words) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long key = (unsigned long long)keys[i];
        unsigned long long mask = (unsigned long long)(32 * words - 1);
        unsigned long long p1 = ((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        unsigned long long p2 = ((key * 0xC2B2AE3D27D4EB4FULL) >> 32) & mask;
        unsigned w1 = bits[p1 >> 5] >> (p1 & 31);
        unsigned w2 = bits[p2 >> 5] >> (p2 & 31);
        match[i] = (w1 & w2 & 1u) ? 1 : 0;
    }
}
