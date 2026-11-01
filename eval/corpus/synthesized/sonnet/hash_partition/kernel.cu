// Per-block histogram in shared memory, merged into the global counts once
// per block. Needs a block size of 256.
extern "C" __global__ void kernel(const long long* keys, int* part, int* counts, long long n, long long p) {
    __shared__ int hist[1024];
    for (long long k = threadIdx.x; k < p; k += blockDim.x) hist[k] = 0;
    __syncthreads();
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned long long h = ((unsigned long long)keys[i] * 0x9E3779B97F4A7C15ULL) >> 32;
        int k = (int)(h % (unsigned long long)p);
        part[i] = k;
        atomicAdd(&hist[k], 1);
    }
    __syncthreads();
    for (long long k = threadIdx.x; k < p; k += blockDim.x)
        if (hist[k] > 0) atomicAdd(&counts[k], hist[k]);
}
