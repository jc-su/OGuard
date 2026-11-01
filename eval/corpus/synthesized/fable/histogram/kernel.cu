// histogram: counts[(key >> shift) & 255] += 1, one radix-sort digit pass.
//
// Each block accumulates a private 256-bin histogram in shared memory (fast
// shared-memory atomics), then flushes its non-zero bins to the global counts
// with one atomicAdd per bin per block.
//
// counts must be zero on entry.

extern "C" __global__ void kernel(const unsigned int* __restrict__ keys,
                                  int* __restrict__ counts,
                                  long long shift,
                                  long long n) {
    __shared__ int local[256];
    for (unsigned j = threadIdx.x; j < 256u; j += blockDim.x) local[j] = 0;
    __syncthreads();

    const unsigned s = (unsigned)shift;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&local[(keys[i] >> s) & 255u], 1);
    }
    __syncthreads();

    for (unsigned j = threadIdx.x; j < 256u; j += blockDim.x) {
        const int c = local[j];
        if (c != 0) atomicAdd(&counts[j], c);
    }
}
