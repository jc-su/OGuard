extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_custkey,
    int64_t* __restrict__ order_counts,
    int64_t n_orders
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t custkey = o_custkey[idx];
        // Safely atomic add since max orders per customer is small 
        // and keys are well-distributed across millions of customers.
        atomicAdd((unsigned long long*)&order_counts[custkey], 1ULL);
    }
}
