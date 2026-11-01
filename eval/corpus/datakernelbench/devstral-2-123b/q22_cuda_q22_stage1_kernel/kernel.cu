extern "C" __global__ void kernel(
    const double* __restrict__ c_acctbal_ptr,
    const bool* __restrict__ in_cntrycode_mask_ptr,
    double* __restrict__ avg_placeholder_ptr,
    int64_t n_customer
) {
    extern __shared__ double sdata[];
    double* block_balance = sdata;
    double* block_count = sdata + blockDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double my_balance = 0.0;
    double my_count = 0.0;
    if (idx < n_customer) {
        bool keep = in_cntrycode_mask_ptr[idx] && (c_acctbal_ptr[idx] > 0.0);
        if (keep) {
            my_balance = c_acctbal_ptr[idx];
            my_count = 1.0;
        }
    }

    block_balance[threadIdx.x] = my_balance;
    block_count[threadIdx.x] = my_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_balance[threadIdx.x] += block_balance[threadIdx.x + s];
            block_count[threadIdx.x] += block_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(avg_placeholder_ptr, block_balance[0]);
        atomicAdd(avg_placeholder_ptr + 1, block_count[0]);
    }
}
