extern "C" __global__ void kernel(
    const int64_t* __restrict__ nation_id,
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    double* __restrict__ out,
    int64_t n_elements,
    int num_nations,
    int num_copies
) {
    extern __shared__ double block_revenue[]; 

    int tid = threadIdx.x;
    int total_bins = num_copies * num_nations;
    // Initialize shared memory
    for (int i = tid; i < total_bins; i += blockDim.x) {
        block_revenue[i] = 0.0;
    }
    __syncthreads();

    // Process elements and accumulate in thread-specific shared memory bank copies
    // to reduce atomicAdd contention.
    int64_t idx = blockIdx.x * blockDim.x + tid;
    if (idx < n_elements) {
        int64_t nid = nation_id[idx];
        if (nid >= 0 && nid < num_nations) {
            double rev = extendedprice[idx] * (1.0 - discount[idx]);
            int copy_idx = tid % num_copies;
            atomicAdd(&block_revenue[copy_idx * num_nations + nid], rev);
        }
    }
    __syncthreads();

    // Reduce copies and write to global memory
    for (int nid = tid; nid < num_nations; nid += blockDim.x) {
        double sum = 0.0;
        for (int c = 0; c < num_copies; ++c) {
            sum += block_revenue[c * num_nations + nid];
        }
        if (sum != 0.0) {
            atomicAdd(&out[nid], sum);
        }
    }
}
