// One thread per row. The arithmetic runs on the unsigned magnitude so that
// -2^63 and the overflow test need no special cases.
extern "C" __global__ void kernel(const long long* values, long long* out, signed char* overflow,
                                  long long n, long long shift) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long v = values[i];
    bool neg = v < 0;
    unsigned long long mag = neg ? 0ull - (unsigned long long)v : (unsigned long long)v;
    if (shift >= 0) {
        unsigned long long p = 1;
        for (long long k = 0; k < shift; ++k) p *= 10ull;
        unsigned long long limit = neg ? 0x8000000000000000ull : 0x7FFFFFFFFFFFFFFFull;
        if (mag > limit / p) {
            out[i] = 0;
            overflow[i] = 1;
        } else {
            unsigned long long r = mag * p;
            out[i] = (long long)(neg ? 0ull - r : r);
            overflow[i] = 0;
        }
    } else {
        unsigned long long q = 1;
        for (long long k = 0; k < -shift; ++k) q *= 10ull;
        unsigned long long qt = mag / q;
        unsigned long long rem = mag - qt * q;
        if (rem * 2 >= q) qt += 1;
        out[i] = (long long)(neg ? 0ull - qt : qt);
        overflow[i] = 0;
    }
}
