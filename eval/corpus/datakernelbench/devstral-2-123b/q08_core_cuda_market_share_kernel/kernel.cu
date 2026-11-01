#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ o_year,
    const double* __restrict__ volume,
    const bool* __restrict__ is_brazil,
    double* __restrict__ sum_brazil,
    double* __restrict__ sum_total,
    int64_t n_elements
) {
    __shared__ double shared_brazil[BLOCK_SIZE];
    __shared__ double shared_total[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_brazil = 0.0;
    double local_total = 0.0;

    if (idx < n_elements) {
        local_total = volume[idx];
        if (is_brazil[idx]) {
            local_brazil = local_total;
        }
    }

    shared_brazil[threadIdx.x] = local_brazil;
    shared_total[threadIdx.x] = local_total;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_brazil[threadIdx.x] += shared_brazil[threadIdx.x + stride];
            shared_total[threadIdx.x] += shared_total[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&sum_brazil[blockIdx.x], shared_brazil[0]);
        atomicAdd(&sum_total[blockIdx.x], shared_total[0]);
    }
}
