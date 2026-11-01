#define MAX_NATIONS 25

extern "C" __global__ void kernel(
    const int64_t* __restrict__ nation_ids,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ out,
    int64_t n_rows,
    int num_nations
) {
    __shared__ double block_agg[MAX_NATIONS];

    // Zero the shared accumulator for this block
    if (threadIdx.x < num_nations) {
        block_agg[threadIdx.x] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_rows) {
        int nid = static_cast<int>(nation_ids[idx]);
        double rev = price[idx] * (1.0 - discount[idx]);
        atomicAdd(&block_agg[nid], rev);
    }
    __syncthreads();

    // Each thread that corresponds to a nation writes its block sum to global output
    if (threadIdx.x < num_nations) {
        atomicAdd(&out[threadIdx.x], block_agg[threadIdx.x]);
    }
}
