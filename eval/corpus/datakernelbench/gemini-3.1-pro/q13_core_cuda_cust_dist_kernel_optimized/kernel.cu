#define HIST_BINS 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    const int64_t* __restrict__ order_counts,
    int64_t* __restrict__ count_frequencies,
    int64_t n_customers
) {
    __shared__ int64_t sh_hist[HIST_BINS];
    
    int tid = threadIdx.x;
    sh_hist[tid] = 0;
    __syncthreads();
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_customers) {
        int64_t custkey = c_custkey[idx];
        int64_t count = order_counts[custkey];
        
        // Vast majority of counts are < 100, which fit nicely in shared memory histogram
        if (count < HIST_BINS) {
            atomicAdd((unsigned long long*)&sh_hist[count], 1ULL);
        } else {
            atomicAdd((unsigned long long*)&count_frequencies[count], 1ULL);
        }
    }
    
    __syncthreads();
    
    // Commit block-local histogram to global memory
    int64_t val = sh_hist[tid];
    if (val > 0) {
        atomicAdd((unsigned long long*)&count_frequencies[tid], (unsigned long long)val);
    }
}
