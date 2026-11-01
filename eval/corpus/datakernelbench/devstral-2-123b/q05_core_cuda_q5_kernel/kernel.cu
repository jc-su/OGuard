extern "C" __global__ void kernel(
    const double* __restrict__ extendedprice_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ nation_id_ptr,
    double* __restrict__ revenue_ptr,
    int64_t num_nations,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double rev = extendedprice_ptr[idx] * (1.0 - discount_ptr[idx]);
        int64_t nation_id = nation_id_ptr[idx];
        if (nation_id >= 0 && nation_id < num_nations) {
            atomicAdd(&revenue_ptr[nation_id], rev);
        }
    }
}
