#define MAX_HIST_BINS 1024

// Kernel to build the histogram of counts for all valid customers.
// Heavily leverages shared memory for the most common order counts (e.g. 0-40) 
// to prevent massive serialization on global atomics, drastically improving speed.
extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey, 
    const int32_t* __restrict__ counts, 
    int32_t* __restrict__ hist, 
    int64_t n, 
    int64_t max_custkey,
    int32_t hist_size
) {
    __shared__ int32_t smem_hist[MAX_HIST_BINS];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < MAX_HIST_BINS; i += blockDim.x) {
        smem_hist[i] = 0;
    }
    __syncthreads();
    
    // Bin assignments per thread
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t c = c_custkey[idx];
        if (c >= 0 && c <= max_custkey) {
            int32_t cnt = counts[c];
            if (cnt < MAX_HIST_BINS) {
                atomicAdd(&smem_hist[cnt], 1);
            } else if (cnt < hist_size) {
                // Fallback for anomalously high counts (rare)
                atomicAdd(&hist[cnt], 1);
            }
        }
    }
    
    __syncthreads();
    
    // Flush shared memory reduction to global memory
    for (int i = threadIdx.x; i < MAX_HIST_BINS; i += blockDim.x) {
        int32_t val = smem_hist[i];
        if (val > 0) {
            atomicAdd(&hist[i], val);
        }
    }
}
