#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkey_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    int64_t* __restrict__ out_keys_ptr,
    double* __restrict__ out_vals_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    __shared__ int64_t block_keys[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    int64_t my_key = 0;

    if (idx < n_elements) {
        my_key = custkey_ptr[idx];
        my_rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
    }

    block_revenue[threadIdx.x] = my_rev;
    block_keys[threadIdx.x] = my_key;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_keys_ptr[blockIdx.x] = block_keys[0];
        out_vals_ptr[blockIdx.x] = block_revenue[0];
    }
}
