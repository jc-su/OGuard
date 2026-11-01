// GPU kernel for scatter_by_offsets.
//
// For each i with flags[i] != 0, writes out[offsets[i]] = i. offsets
// is the exclusive prefix sum of flags, so distinct selected rows
// land at distinct, already-known positions and no atomics are
// needed.

extern "C" __global__ void kernel(const signed char *flags, const int *offsets,
                                   int *out, long long n, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i] != 0) {
        out[offsets[i]] = (int)i;
    }
}
