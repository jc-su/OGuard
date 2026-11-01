// bloom_probe: match[i] = 1 when both hash bits of keys[i] are set in the
// Bloom filter that bloom_build wrote, and 0 otherwise. The bit positions are
// those of bloom_build: the high 32 bits of the key times one of two odd 64-bit
// constants, in wrapping unsigned arithmetic, modulo the 32 * words bits of the
// filter. words is a power of two, so the remainder is a mask. The second word
// is read only when the first bit is set.
#include <cstdint>

extern "C" __global__ void kernel(const int64_t* __restrict__ keys,
                                  const uint32_t* __restrict__ bits,
                                  int8_t* __restrict__ match,
                                  int64_t n,
                                  int64_t words)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint64_t mask = static_cast<uint64_t>(words) * 32 - 1;
        const uint64_t key = static_cast<uint64_t>(keys[i]);
        const uint64_t p1 = ((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        const uint64_t p2 = ((key * 0xC2B2AE3D27D4EB4FULL) >> 32) & mask;
        const bool hit = ((bits[p1 >> 5] >> (p1 & 31)) & 1) && ((bits[p2 >> 5] >> (p2 & 31)) & 1);
        match[i] = hit ? 1 : 0;
    }
}
