#include <cstdint>

extern "C" __global__ void kernel(const int64_t* values, const uint8_t* valid,
                                  uint8_t* out, int64_t n, int64_t lo,
                                  int64_t hi) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    bool keep = false;
    if (i < n) {
        int64_t x = values[i];
        keep = ((valid[i >> 3] >> (i & 7)) & 1) && x >= lo && x <= hi;
    }
    // One ballot covers 32 consecutive rows, which are 4 bytes of the output.
    // Every thread takes part, including those beyond n, whose predicate is 0.
    uint32_t word = __ballot_sync(0xFFFFFFFFu, keep);
    int lane = threadIdx.x & 31;
    int64_t nbytes = (n + 7) >> 3;
    if (lane < 4) {
        int64_t byte = ((int64_t)blockIdx.x * 256 + (threadIdx.x - lane)) / 8 + lane;
        if (byte < nbytes) out[byte] = (uint8_t)((word >> (8 * lane)) & 0xFF);
    }
}
