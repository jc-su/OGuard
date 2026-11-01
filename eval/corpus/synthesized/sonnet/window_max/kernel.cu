// w <= 64: each thread scans its window directly (cached reads).
// w > 64: van Herk / Gil-Werman. A block takes one w-sized segment B, builds
// prefix maxima of segment B and suffix maxima of segment B - 1 in shared
// memory, and out[i] = max(prefix[i], suffix[i - w + 1]). Needs a block size of 256.
#define NEG (-9223372036854775807LL - 1)
extern "C" __global__ void kernel(const long long* values, long long* out, long long n, long long w) {
    int tid = threadIdx.x;
    if (w <= 64) {
        long long stride = (long long)gridDim.x * blockDim.x;
        for (long long i = (long long)blockIdx.x * blockDim.x + tid; i < n; i += stride) {
            long long lo = i - w + 1 < 0 ? 0 : i - w + 1;
            long long m = values[lo];
            for (long long j = lo + 1; j <= i; j++) m = max(m, values[j]);
            out[i] = m;
        }
        return;
    }
    __shared__ long long g[1024], h[1025];
    long long nseg = (n + w - 1) / w;
    for (long long seg = blockIdx.x; seg < nseg; seg += gridDim.x) {
        long long start = seg * w;
        for (int j = tid; j < w; j += 256) {
            g[j] = start + j < n ? values[start + j] : NEG;
            h[j] = seg > 0 ? values[start - w + j] : NEG;
        }
        if (tid == 0) h[w] = NEG;
        __syncthreads();
        for (int d = 1; d < w; d <<= 1) {
            long long gl[4], hl[4];
            int kk = 0;
            for (int j = tid; j < w; j += 256, kk++) {
                long long a = g[j], b = h[j];
                if (j >= d) a = max(a, g[j - d]);
                if (j + d < w) b = max(b, h[j + d]);
                gl[kk] = a; hl[kk] = b;
            }
            __syncthreads();
            kk = 0;
            for (int j = tid; j < w; j += 256, kk++) { g[j] = gl[kk]; h[j] = hl[kk]; }
            __syncthreads();
        }
        for (int j = tid; j < w; j += 256)
            if (start + j < n) out[start + j] = max(g[j], h[j + 1]);
        __syncthreads();
    }
}
