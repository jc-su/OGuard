// Each block accumulates a tile of rows into shared-memory copies of the g
// accumulators and flushes them with one atomic per touched group.
#define TILE 4096
#define MAX_GROUPS 1024

extern "C" __global__ void kernel(const int* group, const long long* value, const unsigned char* valid,
                                  long long* counts, long long* sums, long long n, long long g) {
    __shared__ unsigned long long s_counts[MAX_GROUPS];
    __shared__ unsigned long long s_sums[MAX_GROUPS];
    long long tid = threadIdx.x;
    long long start = (long long)blockIdx.x * TILE;
    long long end = start + TILE < n ? start + TILE : n;
    for (long long k = tid; k < g; k += 256) { s_counts[k] = 0; s_sums[k] = 0; }
    __syncthreads();
    for (long long i = start + tid; i < end; i += 256) {
        if ((valid[i >> 3] >> (i & 7)) & 1) {
            int k = group[i];
            atomicAdd(&s_counts[k], 1ull);
            atomicAdd(&s_sums[k], (unsigned long long)value[i]);
        }
    }
    __syncthreads();
    for (long long k = tid; k < g; k += 256) {
        unsigned long long c = s_counts[k];
        if (c != 0) {
            atomicAdd((unsigned long long*)&counts[k], c);
            atomicAdd((unsigned long long*)&sums[k], s_sums[k]);
        }
    }
}
