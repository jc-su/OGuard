// GPU kernel for starts_with.
//
// match[i] = 1 when string i (offsets[i]:offsets[i+1] in chars)
// starts with the k-byte pattern prefix, else 0. The comparison is
// byte-wise, not code-point aware.

extern "C" __global__ void kernel(const int *offsets, const unsigned char *chars,
                                   const unsigned char *prefix, signed char *match,
                                   long long n, long long k) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int start = offsets[i];
    int end = offsets[i + 1];

    signed char ok = 1;
    if (end - start < k) {
        ok = 0;
    } else {
        for (long long j = 0; j < k; ++j) {
            if (chars[start + j] != prefix[j]) {
                ok = 0;
                break;
            }
        }
    }
    match[i] = ok;
}
