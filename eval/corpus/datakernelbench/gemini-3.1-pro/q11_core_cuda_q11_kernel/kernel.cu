#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ partkey,
    const double* __restrict__ supplycost,
    const double* __restrict__ availqty,
    double* __restrict__ grouped_values,
    double* __restrict__ total_sum,
    int64_t n_elements
) {
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double val = 0.0;
    if (idx < n_elements) {
        val = supplycost[idx] * availqty[idx];
        // Accumulate into the group-by array directly since keys are dense
        atomicAdd(&grouped_values[partkey[idx]], val);
    }
    
    // Block-level reduction for the total sum
    block_sum[threadIdx.x] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(total_sum, block_sum[0]);
    }
}
