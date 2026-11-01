// string_contains: byte substring test of a pattern against each string.
//
// One thread per string. The thread scans the candidate start positions of
// its string, compares the first pattern byte first and only then the rest,
// and stops at the first occurrence. The empty pattern matches every string.
// The pattern is small and read-only, so its loads are served by L1.

extern "C" __global__ void kernel(const int* __restrict__ offsets,
                                  const unsigned char* __restrict__ chars,
                                  const unsigned char* __restrict__ pattern,
                                  signed char* __restrict__ match,
                                  long long n,
                                  long long k)
{
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const long long s = offsets[i];
    const long long e = offsets[i + 1];
    signed char found = 0;
    if (k == 0) {
        found = 1;
    } else if (e - s >= k) {
        const unsigned char p0 = pattern[0];
        const long long last = e - k;
        for (long long pos = s; pos <= last; ++pos) {
            if (chars[pos] != p0) continue;
            long long j = 1;
            while (j < k && chars[pos + j] == pattern[j]) ++j;
            if (j == k) { found = 1; break; }
        }
    }
    match[i] = found;
}
