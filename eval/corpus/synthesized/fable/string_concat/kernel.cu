// One thread per output byte. The row that owns the byte is found by a binary
// search over out_offsets; the byte comes from the first column when its
// position in the row is below that row's first length, and from the second
// column otherwise.
extern "C" __global__ void kernel(const int* a_offsets, const unsigned char* a_chars,
                                  const int* b_offsets, const unsigned char* b_chars,
                                  const int* out_offsets, unsigned char* out_chars, long long n) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = out_offsets[n];
    if (p >= total) return;
    long long lo = 0, hi = n;
    while (hi - lo > 1) {
        long long mid = (lo + hi) >> 1;
        if ((long long)out_offsets[mid] <= p) lo = mid; else hi = mid;
    }
    long long local = p - (long long)out_offsets[lo];
    long long a0 = a_offsets[lo];
    long long la = (long long)a_offsets[lo + 1] - a0;
    if (local < la) out_chars[p] = a_chars[a0 + local];
    else out_chars[p] = b_chars[(long long)b_offsets[lo] + (local - la)];
}
