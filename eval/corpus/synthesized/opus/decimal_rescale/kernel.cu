// decimal_rescale: change the scale of decimal64 values by 10^shift.
//
// shift >= 0: out = values * 10^shift, or out = 0 and overflow = 1 when the product
// leaves the int64 range. For p = 10^shift > 1, p does not divide 2^63, so
// |v| * p fits exactly when |v| <= INT64_MAX / p, for either sign.
// shift < 0: out = values / 10^-shift rounded to nearest, ties away from zero.
// With C division, q = v / d truncates and rem = v % d has the sign of v; the
// result moves one away from zero when 2 * |rem| >= d. |rem| < d <= 10^18, so
// 2 * |rem| does not overflow.

extern "C" __global__ void kernel(const long long* __restrict__ values,
                                  long long* __restrict__ out,
                                  signed char* __restrict__ overflow,
                                  long long n, long long shift)
{
    const long long k = shift >= 0 ? shift : -shift;
    long long p = 1;
    for (long long e = 0; e < k; ++e) p *= 10;
    const long long limit = 0x7fffffffffffffffll / p;

    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long v = values[i];
        if (shift >= 0) {
            if (shift == 0 || (v >= -limit && v <= limit)) {
                out[i] = v * p;
                overflow[i] = 0;
            } else {
                out[i] = 0;
                overflow[i] = 1;
            }
        } else {
            long long q = v / p;
            const long long rem = v % p;
            const long long mag = rem < 0 ? -rem : rem;
            if (2 * mag >= p) q += v < 0 ? -1 : 1;
            out[i] = q;
            overflow[i] = 0;
        }
    }
}
