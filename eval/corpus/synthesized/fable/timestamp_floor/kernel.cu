// One thread per row. The remainder is taken on a nonnegative magnitude, so a
// timestamp before 1970 rounds toward negative infinity.
extern "C" __global__ void kernel(const long long* ts, long long* out, long long n, long long unit) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long t = ts[i];
    if (t >= 0) {
        out[i] = t - t % unit;
    } else {
        long long r = (-t) % unit;
        out[i] = r == 0 ? t : t - (unit - r);
    }
}
