// timestamp_floor: out[i] = largest multiple of unit that is <= ts[i] (rounds toward
// negative infinity, also before 1970).
//
// C division truncates toward zero, so the quotient is lowered by one when the
// remainder is negative. |ts| <= 2^62 and unit <= 2^40, so q * unit stays in range.

extern "C" __global__ void kernel(const long long* __restrict__ ts,
                                  long long* __restrict__ out,
                                  long long n, long long unit)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long t = ts[i];
        long long q = t / unit;
        if (t - q * unit < 0) --q;
        out[i] = q * unit;
    }
}
