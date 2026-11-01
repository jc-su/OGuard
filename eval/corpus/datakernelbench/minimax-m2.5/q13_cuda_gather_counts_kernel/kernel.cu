extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkeys,
    const int64_t* __restrict__ histogram,
    int64_t* __restrict__ counts,
    int64_t n_customers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_customers) {
        int64_t custkey = custkeys[idx];
        counts[idx] = histogram[custkey];
    }
}
