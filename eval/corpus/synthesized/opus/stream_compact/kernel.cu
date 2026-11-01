// Stream compaction: write the values of the kept rows to the front of out.
#include <cstdint>

extern "C" __global__ void kernel(const int64_t* values, const int8_t* keep, int64_t* out,
                                  int32_t* count, int64_t n) {
    const unsigned lane = threadIdx.x & 31u;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    // i - lane is the first row of the warp and is equal in all 32 lanes, so all
    // lanes run the same iterations and the full-mask warp intrinsics are valid.
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i - static_cast<int64_t>(lane) < n; i += stride) {
        const bool flag = i < n && keep[i] != 0;
        const unsigned mask = __ballot_sync(0xFFFFFFFFu, flag);
        // One atomic per warp reserves the output slots of all kept lanes.
        int32_t base = 0;
        if (lane == 0 && mask != 0) {
            base = atomicAdd(count, __popc(mask));
        }
        base = __shfl_sync(0xFFFFFFFFu, base, 0);
        if (flag) {
            out[base + __popc(mask & ((1u << lane) - 1u))] = values[i];
        }
    }
}
