// One thread per left row: binary search for the last right row at or before
// the left value, then apply the tolerance.
extern "C" __global__ void kernel(const long long* left, const long long* right, int* out,
                                  long long n, long long r, long long tolerance) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long x = left[i];
    long long lo = 0, hi = r;
    while (lo < hi) {
        long long mid = (lo + hi) >> 1;
        if (right[mid] <= x) lo = mid + 1; else hi = mid;
    }
    long long j = lo - 1;
    int res = -1;
    if (j >= 0 && x - right[j] <= tolerance) res = (int)j;
    out[i] = res;
}
