extern "C" __global__ void kernel(long long* values, long long* out, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx == 0) {
            out[idx] = values[idx];
        } else {
            out[idx] = values[idx] - values[idx - 1];
        }
    }
}
