// One thread per byte of the validity bitmap: it parses the eight strings of
// that byte in turn and writes the byte once, so no atomics on the bitmap are
// needed. The magnitude accumulates in unsigned arithmetic with the overflow
// test done before each multiply-add, so that -9223372036854775808 parses and
// 9223372036854775808 does not.
extern "C" __global__ void kernel(const int* offsets, const unsigned char* chars, long long* out,
                                  unsigned char* valid, long long n) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nbytes = (n + 7) / 8;
    if (b >= nbytes) return;
    unsigned int bits = 0;
    for (int k = 0; k < 8; ++k) {
        long long i = b * 8 + k;
        if (i >= n) break;
        long long p = offsets[i], end = offsets[i + 1];
        bool ok = false;
        long long value = 0;
        if (p < end) {
            bool neg = false;
            unsigned char c = chars[p];
            if (c == '+' || c == '-') { neg = c == '-'; ++p; }
            if (p < end) {
                unsigned long long limit = neg ? 0x8000000000000000ull : 0x7FFFFFFFFFFFFFFFull;
                unsigned long long mag = 0;
                ok = true;
                while (p < end) {
                    c = chars[p];
                    if (c < '0' || c > '9') { ok = false; break; }
                    unsigned long long d = c - '0';
                    if (mag > (limit - d) / 10ull) { ok = false; break; }
                    mag = mag * 10ull + d;
                    ++p;
                }
                if (ok) value = (long long)(neg ? 0ull - mag : mag);
            }
        }
        out[i] = value;
        if (ok) bits |= 1u << k;
    }
    valid[b] = (unsigned char)bits;
}
