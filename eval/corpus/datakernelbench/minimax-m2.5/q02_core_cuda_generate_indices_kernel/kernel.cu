extern "C" __global__ void kernel(
    int64_t* __restrict__ out_ptr,
    int64_t n_elements,
    int64_t k
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k && idx < n_elements) {
        out_ptr[idx] = idx;
    }
}
