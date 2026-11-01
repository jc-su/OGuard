// GPU kernel for histogram.
//
// Counts keys by the byte (key >> shift) & 255, one pass of a radix
// sort. Each block keeps a private histogram in shared memory
// (indexed 1:1 by thread id, since BLOCK == 256 bins) and flushes it
// to the global counts with one atomic add per bin per block, instead
// of one atomic add per key. The host caps the grid at MAX_BLOCKS and
// each thread strides over multiple keys, so the flush cost does not
// grow with n.
//
// Precondition: the caller zeroes counts before each launch.

const int NUM_BINS = 256;

extern "C" __global__ void kernel(const unsigned int *keys, int *counts,
                                   long long shift, long long n) {
    __shared__ int local[NUM_BINS];

    int tid = threadIdx.x;
    local[tid] = 0;
    __syncthreads();

    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    while (i < n) {
        int b = (int)((keys[i] >> shift) & 255U);
        atomicAdd(&local[b], 1);
        i += stride;
    }
    __syncthreads();

    atomicAdd(&counts[tid], local[tid]);
}
