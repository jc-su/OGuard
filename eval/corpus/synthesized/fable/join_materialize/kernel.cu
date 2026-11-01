// One thread per output pair. The probe row that owns the pair is found by a
// binary search over offsets, and its first match in build by a binary search
// over build, so a probe row with many matches is still balanced.
extern "C" __global__ void kernel(const long long* build, const long long* probe, const int* offsets,
                                  int* out_build, int* out_probe, long long b, long long p) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = offsets[p];
    if (q >= total) return;
    long long lo = 0, hi = p;
    while (hi - lo > 1) {
        long long mid = (lo + hi) >> 1;
        if ((long long)offsets[mid] <= q) lo = mid; else hi = mid;
    }
    long long i = lo;
    long long key = probe[i];
    long long flo = 0, fhi = b;
    while (flo < fhi) {
        long long mid = (flo + fhi) >> 1;
        if (build[mid] < key) flo = mid + 1; else fhi = mid;
    }
    out_build[q] = (int)(flo + (q - (long long)offsets[i]));
    out_probe[q] = (int)i;
}
