extern "C" __global__ void kernel(double* values, char* mask, double* result, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx] != 0) {
            atomicAdd(result, values[idx]);
        }
    }
}
