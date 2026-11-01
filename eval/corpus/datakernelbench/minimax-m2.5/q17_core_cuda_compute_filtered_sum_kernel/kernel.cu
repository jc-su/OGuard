#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ part_idx,
    const double* __restrict__ quantity,
    const double* __restrict__ price,
    const double* __restrict__ threshold,
    double* __restrict__ total,
    int64_t n
) {
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double my_val = 0.0;
    if (tid < n) {
        int64_t p = part_idx[tid];
        double q = quantity[tid];
        if (q < threshold[p]) {
            my_val = price[tid];
        }
    }
    block_sum[threadIdx.x] = my_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(total, block_sum[0]);
    }
}
