extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys_sorted,
    const double* __restrict__ values_sorted,
    int64_t* __restrict__ keys_out,
    double* __restrict__ values_out,
    int32_t* __restrict__ unique_count,
    int64_t n
) {
    // Grid-stride loop to handle all elements
    for (int64_t base = blockIdx.x * blockDim.x; base < n; base += blockDim.x * gridDim.x) {
        int64_t i = base + threadIdx.x;
        if (i >= n) continue;
        
        // Check if this is the start of a new key run
        // Either at beginning of array or key differs from previous
        if (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) {
            int64_t key = keys_sorted[i];
            double sum = values_sorted[i];
            
            // Accumulate all values for this key
            int64_t j = i + 1;
            while (j < n && keys_sorted[j] == key) {
                sum += values_sorted[j];
                j++;
            }
            
            // Atomically claim output position and write
            int out_idx = atomicAdd(unique_count, 1);
            keys_out[out_idx] = key;
            values_out[out_idx] = sum;
        }
    }
}
