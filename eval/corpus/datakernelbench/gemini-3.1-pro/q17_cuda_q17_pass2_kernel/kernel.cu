#define BLOCK_SIZE 1024

// Pass 2: Iterate over lineitems again, re-check part map, compute threshold
// based on avg(qty), and conditionally accumulate block-wise revenue.
extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const int32_t* __restrict__ part_map,
    int64_t max_partkey,
    const double* __restrict__ sum_qty,
    const int32_t* __restrict__ cnt_qty,
    double* __restrict__ out_total,
    int64_t n_lineitems
) {
    __shared__ double block_rev[BLOCK_SIZE];
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (i < n_lineitems) {
        int64_t pk = l_partkey[i];
        if (pk >= 0 && pk <= max_partkey) {
            int32_t p_idx = part_map[pk];
            if (p_idx != -1) {
                double qty = l_quantity[i];
                double s_qty = sum_qty[p_idx];
                int32_t c_qty = cnt_qty[p_idx];
                
                if (c_qty > 0) {
                    double avg_qty = s_qty / (double)c_qty;
                    if (qty < 0.2 * avg_qty) {
                        my_rev = l_extendedprice[i];
                    }
                }
            }
        }
    }
    
    block_rev[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_rev[threadIdx.x] += block_rev[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Global atomic add
    if (threadIdx.x == 0 && block_rev[0] > 0.0) {
        atomicAdd(out_total, block_rev[0]);
    }
}
