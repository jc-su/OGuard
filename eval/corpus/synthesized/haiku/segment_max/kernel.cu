extern "C" __global__ void kernel(int* offsets, long long* values, long long* maxes, int g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < g) {
        int start = offsets[idx];
        int end = offsets[idx + 1];

        // Initialize with smallest int64
        long long max_val = LLONG_MIN;

        // Find max in this segment
        for (int i = start; i < end; i++) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }

        maxes[idx] = max_val;
    }
}
