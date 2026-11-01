// Hash partition: part[i] = h(keys[i]) % p and counts[k] = rows in partition k.
#include <cstdint>

constexpr int kMaxPartitions = 1024;
constexpr uint64_t kMultiplier = 0x9E3779B97F4A7C15ull;

extern "C" __global__ void kernel(const int64_t* keys, int32_t* part, int32_t* counts,
                                  int64_t n, int64_t p) {
    // Each block counts its rows in a shared histogram and adds the nonzero bins to
    // counts once at the end.
    __shared__ int32_t hist[kMaxPartitions];
    for (int64_t k = threadIdx.x; k < p; k += blockDim.x) hist[k] = 0;
    __syncthreads();

    // h(x) < 2^32 and p <= 1024, so the remainder fits in 32-bit unsigned arithmetic.
    const uint32_t parts = static_cast<uint32_t>(p);
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const uint32_t h = static_cast<uint32_t>((static_cast<uint64_t>(keys[i]) * kMultiplier) >> 32);
        const uint32_t k = h % parts;
        part[i] = static_cast<int32_t>(k);
        atomicAdd(&hist[k], 1);
    }
    __syncthreads();

    for (int64_t k = threadIdx.x; k < p; k += blockDim.x) {
        const int32_t c = hist[k];
        if (c != 0) atomicAdd(&counts[k], c);
    }
}
