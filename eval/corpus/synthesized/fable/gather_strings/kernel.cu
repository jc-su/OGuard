// One thread per output byte. The owning row is found by a binary search over
// out_offsets, so the work is balanced for any string lengths and the writes
// are coalesced.
extern "C" __global__ void kernel(const int* offsets, const unsigned char* chars, const int* idx,
                                  const int* out_offsets, unsigned char* out_chars,
                                  long long n, long long m) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = out_offsets[m];
    if (p >= total) return;
    long long lo = 0, hi = m;
    while (hi - lo > 1) {
        long long mid = (lo + hi) >> 1;
        if ((long long)out_offsets[mid] <= p) lo = mid; else hi = mid;
    }
    long long row = idx[lo];
    long long src = (long long)offsets[row] + (p - (long long)out_offsets[lo]);
    out_chars[p] = chars[src];
}
