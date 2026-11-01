#include <cstdint>

extern "C" __global__ void kernel(const int64_t* __restrict__ keys, const uint32_t* __restrict__ bits,
                                  int8_t* __restrict__ match, int64_t n, int64_t words) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint64_t k = (uint64_t)keys[i];
    uint64_t mask = 32 * (uint64_t)words - 1;  // words is a power of two
    uint64_t p1 = ((k * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
    uint64_t p2 = ((k * 0xC2B2AE3D27D4EB4FULL) >> 32) & mask;
    match[i] = ((bits[p1 >> 5] >> (p1 & 31)) & (bits[p2 >> 5] >> (p2 & 31)) & 1u) ? 1 : 0;
  }
}
