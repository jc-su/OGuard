extern "C" __global__ void kernel(int* idx, double* values, double* out, long long m, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int out_idx = idx[i];
        if (out_idx >= 0 && out_idx < n) {
            atomicAdd(&out[out_idx], values[i]);
        }
    }
}
