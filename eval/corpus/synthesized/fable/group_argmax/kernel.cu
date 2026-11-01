// One thread per row: a compare-and-swap loop on best[group] that keeps the
// row with the larger value, or the smaller row on a tie. The loop ends as
// soon as the published row is at least as good as this one.
extern "C" __global__ void kernel(const int* group, const long long* value, int* best, long long n, long long g) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = group[i];
    long long v = value[i];
    int cur = best[k];
    while (true) {
        if (cur >= 0) {
            long long cv = value[cur];
            if (cv > v || (cv == v && cur < i)) break;
        }
        int old = atomicCAS(&best[k], cur, (int)i);
        if (old == cur) break;
        cur = old;
    }
}
