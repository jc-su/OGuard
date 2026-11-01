extern "C" __global__ void kernel(int64_t* out, int64_t k) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        out[idx] = idx;
    }
}
