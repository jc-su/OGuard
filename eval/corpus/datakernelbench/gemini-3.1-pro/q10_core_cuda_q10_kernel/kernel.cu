extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkey_ptr,
    const double* __restrict__ extprice_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ rev_out,
    bool* __restrict__ present_out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = custkey_ptr[idx];
        double price = extprice_ptr[idx];
        double discount = discount_ptr[idx];
        double rev = price * (1.0 - discount);

        atomicAdd(&rev_out[key], rev);
        present_out[key] = true;
    }
}
