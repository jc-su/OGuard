extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys_in,
    const double* __restrict__ values_in,
    int64_t* __restrict__ keys_out,
    double* __restrict__ values_out,
    int32_t* __restrict__ out_count,
    int32_t n_in,
    double threshold
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_in) return;
    
    if (values_in[idx] > threshold) {
        int out_idx = atomicAdd(out_count, 1);
        keys_out[out_idx] = keys_in[idx];
        values_out[out_idx] = values_in[idx];
    }
}
