// Lower bound, then upper bound searched from the lower bound.
extern "C" __global__ void kernel(const long long* build, const long long* probe,
                                  int* counts, long long b, long long p) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < p; i += stride) {
        long long x = probe[i];
        long long lo = 0, hi = b;
        while (lo < hi) {
            long long mid = (lo + hi) >> 1;
            if (build[mid] < x) lo = mid + 1; else hi = mid;
        }
        long long first = lo;
        hi = b;
        while (lo < hi) {
            long long mid = (lo + hi) >> 1;
            if (build[mid] <= x) lo = mid + 1; else hi = mid;
        }
        counts[i] = (int)(lo - first);
    }
}
