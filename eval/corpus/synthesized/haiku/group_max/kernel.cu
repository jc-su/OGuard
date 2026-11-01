extern "C" __global__ void kernel(int* group, long long* value, long long* best, long long n, int capacity) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int g = group[idx];
        long long v = value[idx];
        if (g >= 0 && g < capacity) {
            // Atomic max
            atomicMax(&best[g], v);
        }
    }
}
