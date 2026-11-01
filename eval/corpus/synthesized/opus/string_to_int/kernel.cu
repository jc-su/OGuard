// string_to_int: parse string i as an optional sign followed by one or more ASCII
// digits (any number of leading zeros) into int64; out[i] = value and valid bit i = 1
// when it parses and fits, else out[i] = 0 and the bit stays 0.
//
// The magnitude accumulates in uint64 and is checked before each step against the
// limit 2^63 - 1 (or 2^63 for a minus sign): mag * 10 + d <= limit exactly when
// mag <= (limit - d) / 10. Parsing stops at the first invalid byte. Each warp
// parses 32 consecutive rows starting at a multiple of 32 and assembles their valid
// bits with a ballot; lanes 0..3 store the four bitmap bytes, so each byte has one
// writer and the bits at positions n and above are 0.

extern "C" __global__ void kernel(const int* __restrict__ offsets,
                                  const unsigned char* __restrict__ chars,
                                  long long* __restrict__ out,
                                  unsigned char* __restrict__ valid,
                                  long long n)
{
    const int lane = threadIdx.x & 31;
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long long num_warps = ((long long)gridDim.x * blockDim.x) >> 5;
    const long long out_bytes = (n + 7) >> 3;

    for (long long base = warp * 32; base < n; base += num_warps * 32) {
        const long long i = base + lane;
        bool ok = false;
        if (i < n) {
            long long p = offsets[i];
            const long long end = offsets[i + 1];
            bool neg = false;
            if (p < end && (chars[p] == '+' || chars[p] == '-')) {
                neg = chars[p] == '-';
                ++p;
            }
            ok = p < end;
            const unsigned long long limit = neg ? 0x8000000000000000ull : 0x7fffffffffffffffull;
            unsigned long long mag = 0;
            for (; ok && p < end; ++p) {
                const unsigned int d = (unsigned int)chars[p] - '0';
                if (d > 9 || mag > (limit - d) / 10) ok = false;
                else mag = mag * 10 + d;
            }
            out[i] = ok ? (long long)(neg ? 0ull - mag : mag) : 0;
        }
        const unsigned int word = __ballot_sync(0xffffffffu, ok);
        const long long byte = (base >> 3) + lane;
        if (lane < 4 && byte < out_bytes) valid[byte] = (unsigned char)(word >> (8 * lane));
    }
}
