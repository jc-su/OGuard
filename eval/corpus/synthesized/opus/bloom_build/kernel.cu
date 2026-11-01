// bloom_build: set two hash bits of each key in a Bloom filter of 32 * words
// bits, where bit p is bit p % 32 of bits[p / 32]. The bit positions of a key
// are h1 % (32 * words) and h2 % (32 * words), with h1 and h2 the high 32 bits
// of the key times one of two odd 64-bit constants, in wrapping unsigned
// arithmetic. words is a power of two, so the remainder is a mask.
#include <cstdint>

extern "C" __global__ void kernel(const int64_t* __restrict__ keys,
                                  uint32_t* __restrict__ bits,
                                  int64_t n,
                                  int64_t words)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint64_t mask = static_cast<uint64_t>(words) * 32 - 1;
        const uint64_t key = static_cast<uint64_t>(keys[i]);
        const uint64_t p1 = ((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        const uint64_t p2 = ((key * 0xC2B2AE3D27D4EB4FULL) >> 32) & mask;
        atomicOr(&bits[p1 >> 5], 1u << (p1 & 31));
        atomicOr(&bits[p2 >> 5], 1u << (p2 & 31));
    }
}
