// filter_select: sel[0:count] = row ids i with quantity[i] < threshold.
//
// Warp-aggregated compaction. Each thread walks the rows with a grid stride.
// On a selected row, the thread reads the active mask, the lowest active lane
// does a single atomicAdd on count for all active lanes, and every active lane
// writes its row id at base + (number of active lanes below it). Under
// independent thread scheduling the active mask can cover a subset of the
// lanes that hold a selected row; each subset then reserves its own slots,
// which is still correct.
//
// The store sits inside the row loop, whose condition i < n bounds the stored
// row id on every path to the store. The output position is formed from base
// as an unsigned value and compared against n before the store, so every store
// stays inside sel (n slots) whatever count holds on entry. Under the contract
// (count zero on entry) the comparison always holds, because at most n rows
// are selected.
//
// count must be zero on entry.

extern "C" __global__ void kernel(const double* __restrict__ quantity,
                                  double threshold,
                                  int* __restrict__ sel,
                                  int* __restrict__ count,
                                  long long n) {
    const unsigned lanemask_lt = (1u << (threadIdx.x & 31u)) - 1u;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (quantity[i] < threshold) {
            const unsigned active = __activemask();
            const unsigned rank = __popc(active & lanemask_lt);
            int base = 0;
            if (rank == 0u) base = atomicAdd(count, __popc(active));
            base = __shfl_sync(active, base, __ffs(active) - 1);
            const long long pos = (unsigned)base + rank;
            if (pos < n) sel[pos] = (int)i;
        }
    }
}
