extern "C" __global__ void kernel(double* values, double* bounds, int* counts, long long n, int b) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = values[idx];
        // Count how many bounds <= val
        int count = 0;
        for (int i = 0; i < b; i++) {
            if (bounds[i] <= val) {
                count++;
            }
        }
        atomicAdd(&counts[count], 1);
    }
}
