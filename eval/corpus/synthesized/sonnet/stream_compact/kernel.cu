// Warp-aggregated stream compaction: one atomic per warp that has selected
// rows. All threads of a block run the same number of iterations, so the
// full-warp ballot and shuffle are reached by all 32 lanes. Needs a block
// size that is a multiple of 32.
extern "C" __global__ void kernel(const long long* values, const signed char* keep,
                                  long long* out, int* count, long long n) {
    unsigned lane = threadIdx.x & 31;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long base = (long long)blockIdx.x * blockDim.x; base < n; base += stride) {
        long long i = base + threadIdx.x;
        bool pred = i < n && keep[i] != 0;
        unsigned mask = __ballot_sync(0xffffffffu, pred);
        if (mask) {
            int pos = 0;
            if (lane == 0) pos = atomicAdd(count, __popc(mask));
            pos = __shfl_sync(0xffffffffu, pos, 0);
            if (pred) out[pos + __popc(mask & ((1u << lane) - 1u))] = values[i];
        }
    }
}
