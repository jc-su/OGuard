// Kernel to count occurrences of each order customer key.
// Passing the boolean mask directly into the kernel prevents the need to
// eagerly materialize a filtered DataFrame in memory beforehand, saving
// significant allocation time and memory bandwidth.
extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_custkey, 
    const bool* __restrict__ mask, 
    int32_t* __restrict__ counts, 
    int64_t n, 
    int64_t max_custkey
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx]) {
            int64_t cust = o_custkey[idx];
            if (cust >= 0 && cust <= max_custkey) {
                atomicAdd(&counts[cust], 1);
            }
        }
    }
}
