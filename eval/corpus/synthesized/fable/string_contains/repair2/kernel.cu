// string_contains: byte substring test of a pattern against each string.
//
// One thread per string. The thread walks the candidate start positions of
// its string and compares the pattern byte by byte in one flat loop: a
// match of byte j advances j, a mismatch moves to the next position and
// resets j, and j reaching k reports the match. The empty pattern matches
// every string. The pattern is small and read-only, so its loads are served
// by L1.
//
// Revision: the nested loop and the min/max clamp of the pattern index are
// gone; the pattern index j and the byte index q = position + j are tested
// against [0, k) and [0, offsets[n]) right before their loads. Both pass
// through an empty asm statement so that the compiler keeps those tests on
// the very register that forms the address. Under the task's contract the
// tests always pass.

#define THREADS 256

__device__ __forceinline__ long long opaque(long long x)
{
    asm("" : "+l"(x));
    return x;
}

extern "C" __global__ void kernel(const int* __restrict__ offsets,
                                  const unsigned char* __restrict__ chars,
                                  const unsigned char* __restrict__ pattern,
                                  signed char* __restrict__ match,
                                  long long n,
                                  long long k)
{
    const long long i = (long long)blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;
    const long long s = offsets[i];
    const long long e = offsets[i + 1];
    const long long total = offsets[n];
    signed char found = 0;
    if (k == 0) {
        found = 1;
    } else {
        const long long last = e - k;  // last candidate start position
        long long pos = s;
        long long j = 0;
        while (pos <= last) {
            j = opaque(j);
            if (j < 0 || j >= k) break;  // never under the contract
            const long long q = opaque(pos + j);
            if (q < 0 || q >= total) break;  // never under the contract
            if (chars[q] == pattern[j]) {
                ++j;
                if (j == k) { found = 1; break; }
            } else {
                ++pos;
                j = 0;
            }
        }
    }
    match[i] = found;
}
