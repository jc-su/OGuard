// string_contains: byte substring test of a pattern against each string.
//
// One thread per string. The thread scans the candidate start positions of
// its string, compares the pattern byte by byte, and stops at the first
// occurrence. The empty pattern matches every string. The pattern is small
// and read-only, so its loads are served by L1.
//
// Revision: the pattern index is clamped to [0, k - 1] and every byte
// position q of `chars` is tested against [0, total), with
// total = offsets[n] the size of `chars`, before the loads. Inside the loops
// 0 <= j < k and s <= q < e <= total, so neither changes the result; they
// state the bound of each access for the memory checker, which does not
// carry loop invariants.

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
    const long long total = offsets[n];
    signed char found = 0;
    if (k == 0) {
        found = 1;
    } else if (e - s >= k) {
        const long long last = e - k;
        for (long long pos = s; pos <= last; ++pos) {
            long long j = 0;
            while (j < k) {
                const long long jc = j < 0 ? 0 : (j < k - 1 ? j : k - 1);  // == j
                const long long q = pos + jc;
                if (q < 0 || q >= total) break;
                if (chars[q] != pattern[jc]) break;
                j = jc + 1;
            }
            if (j == k) { found = 1; break; }
        }
    }
    match[i] = found;
}
