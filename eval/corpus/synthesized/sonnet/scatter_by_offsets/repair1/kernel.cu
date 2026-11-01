// GPU kernel for scatter_by_offsets.
//
// For each i with flags[i] != 0, writes out[offsets[i]] = i. offsets
// is the exclusive prefix sum of flags, so distinct selected rows
// land at distinct, already-known positions and no atomics are
// needed.
//
// offsets is an input array, not a scalar with a declared value
// range, so the checker cannot take "each used value lands in
// [0, total)" for granted from the array's dtype alone: the store is
// guarded by an explicit two-sided bounds check on the position read
// from offsets before it runs. Under the stated contract, offsets is
// the true exclusive prefix sum of flags, so the guard never rejects
// a position for a row with flags[i] != 0.

extern "C" __global__ void kernel(const signed char *flags, const int *offsets,
                                   int *out, long long n, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i] != 0) {
        int pos = offsets[i];
        if (pos >= 0 && pos < total) {
            out[pos] = (int)i;
        }
    }
}
