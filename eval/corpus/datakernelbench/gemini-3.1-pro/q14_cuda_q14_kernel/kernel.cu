#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const bool* __restrict__ promo,
    double* __restrict__ out_promo_rev,
    double* __restrict__ out_total_rev,
    int64_t n
) {
    __shared__ double block_promo[BLOCK_SIZE];
    __shared__ double block_total[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    double p_rev = 0.0;
    double t_rev = 0.0;

    // Grid stride loop ensures kernel works securely with arbitrary n elements 
    for (int64_t i = idx; i < n; i += stride) {
        double p = price[i];
        double d = discount[i];
        bool pr = promo[i];
        double rev = p * (1.0 - d);
        t_rev += rev;
        if (pr) {
            p_rev += rev;
        }
    }

    block_promo[threadIdx.x] = p_rev;
    block_total[threadIdx.x] = t_rev;
    __syncthreads();

    // Shared memory tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Single global atomic addition per block
    if (threadIdx.x == 0) {
        atomicAdd(out_promo_rev, block_promo[0]);
        atomicAdd(out_total_rev, block_total[0]);
    }
}
