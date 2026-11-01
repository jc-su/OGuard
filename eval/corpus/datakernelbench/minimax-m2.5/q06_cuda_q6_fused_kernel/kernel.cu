#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    // Shared memory for block-level reduction
    __shared__ double block_revenue[BLOCK_SIZE];
    
    // Calculate global thread index
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Per-thread partial revenue accumulator
    double my_rev = 0.0;
    
    // Each thread processes one element (coalesced memory access)
    if (idx < n_elements) {
        // Load data with restrict pointers for better compiler optimization
        const int32_t sd = shipdate_ptr[idx];
        const double disc = discount_ptr[idx];
        const double qty = quantity_ptr[idx];
        
        // Predicate evaluation in registers
        const bool keep = (sd >= date_lo) && (sd < date_hi)
                       && (disc >= disc_lo) && (disc <= disc_hi)
                       && (qty < qty_hi);
        
        // Compute revenue only for matching rows
        if (keep) {
            my_rev = price_ptr[idx] * disc;
        }
    }
    
    // Store per-thread result to shared memory
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Block-level tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block atomically adds to global output
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}
