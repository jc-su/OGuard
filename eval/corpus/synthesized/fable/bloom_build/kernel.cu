// bloom_build: for each key, set bits h1 % (32 * words) and h2 % (32 * words)
// of a Bloom filter, where bit p is bit p % 32 of bits[p / 32] and
// h1 = (key * 0x9E3779B97F4A7C15) >> 32, h2 = (key * 0xC2B2AE3D27D4EB4F) >> 32
// in wrapping unsigned 64-bit arithmetic. words is a power of two.
extern "C" __global__ void kernel(const long long* keys,
                                  unsigned* bits,
                                  long long n,
                                  long long words) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long key = (unsigned long long)keys[i];
        unsigned long long mask = (unsigned long long)(32 * words - 1);
        unsigned long long p1 = ((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        unsigned long long p2 = ((key * 0xC2B2AE3D27D4EB4FULL) >> 32) & mask;
        atomicOr(&bits[p1 >> 5], 1u << (p1 & 31));
        atomicOr(&bits[p2 >> 5], 1u << (p2 & 31));
    }
}
