// Popcount per byte, warp and block reduction, one atomic per block.
// Needs a block size of 256.
extern "C" __global__ void kernel(const unsigned char* valid, long long* count, long long n) {
    __shared__ long long sh[8];
    long long nbytes = (n + 7) / 8;
    int rem = (int)(n & 7);
    long long stride = (long long)gridDim.x * blockDim.x;
    long long acc = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nbytes; i += stride) {
        unsigned v = valid[i];
        if (i == nbytes - 1 && rem != 0) acc += rem - __popc(v & ((1u << rem) - 1u));
        else acc += 8 - __popc(v);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        long long total = 0;
        for (int w = 0; w < 8; w++) total += sh[w];
        if (total > 0) atomicAdd((unsigned long long*)count, (unsigned long long)total);
    }
}
