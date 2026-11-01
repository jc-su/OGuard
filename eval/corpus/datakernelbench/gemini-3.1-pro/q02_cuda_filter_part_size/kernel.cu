extern "C" __global__ void kernel(
    const int64_t* __restrict__ p_size,
    int64_t size_val,
    bool* __restrict__ out_mask,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_mask[idx] = (p_size[idx] == size_val);
    }
}
