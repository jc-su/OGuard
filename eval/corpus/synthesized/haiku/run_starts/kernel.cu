extern "C" __global__ void kernel(long long* values, char* flags, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx == 0) {
            flags[idx] = 1;
        } else {
            flags[idx] = (values[idx] != values[idx - 1]) ? 1 : 0;
        }
    }
}
