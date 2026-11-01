#include <cstdint>

extern "C" __global__ void kernel(const int64_t* __restrict__ part,
                                const int64_t* __restrict__ supp,
                                uint64_t* __restrict__ out,
                                int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint64_t pk = static_cast<uint64_t>(part[idx]);
        uint64_t sk = static_cast<uint64_t>(supp[idx]);
        out[idx] = (pk << 32) | sk;
    }
}
