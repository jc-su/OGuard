// filter_select: sel[0:count] = row ids i with quantity[i] < threshold.
//
// Warp-aggregated compaction. Each warp ballots the predicate, lane 0 does a
// single atomicAdd on count for the whole warp, and every selected lane writes
// its row id at base + (number of selected lanes below it). The loop bound is
// rounded up to a multiple of the warp size so that every ballot is a full warp.
//
// The output position is formed from base as an unsigned value and compared
// against n before the store, so every store stays inside sel (n slots)
// whatever count holds on entry. Under the contract (count zero on entry) the
// comparison always holds, because at most n rows are selected.
//
// count must be zero on entry.

extern "C" __global__ void kernel(const double* __restrict__ quantity,
                                  double threshold,
                                  int* __restrict__ sel,
                                  int* __restrict__ count,
                                  long long n) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const long long n_round = (n + 31) & ~31LL;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n_round; i += stride) {
        const bool pred = (i < n) && (quantity[i] < threshold);
        const unsigned ballot = __ballot_sync(0xffffffffu, pred);
        if (ballot == 0u) continue;  // uniform across the warp
        int base = 0;
        if (lane == 0u) base = atomicAdd(count, __popc(ballot));
        base = __shfl_sync(0xffffffffu, base, 0);
        if (pred) {
            const long long pos = (unsigned)base + (unsigned)__popc(ballot & lanemask_lt);
            if (pos < n) sel[pos] = (int)i;
        }
    }
}
