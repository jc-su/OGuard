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
    __shared__ double block_revenue[BLOCK_SIZE];
    
    // Grid-stride loop indexing
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    double my_rev = 0.0;
    
    // Late materialization / predicate pushdown: 
    // Load expensive columns only when previous predicates succeed.
    for (int64_t i = idx; i < n_elements; i += stride) {
        int32_t sd = shipdate_ptr[i];
        if (sd >= date_lo && sd < date_hi) {
            double disc = discount_ptr[i];
            if (disc >= disc_lo && disc <= disc_hi) {
                double qty = quantity_ptr[i];
                if (qty < qty_hi) {
                    my_rev += price_ptr[i] * disc;
                }
            }
        }
    }
    
    // Block-level reduction in shared memory
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Single atomicAdd per block
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}
