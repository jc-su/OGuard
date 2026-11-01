extern "C" __global__ void kernel(
    const int64_t* __restrict__ order_custkeys,
    int64_t* __restrict__ histogram,
    int64_t n_orders
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t custkey = order_custkeys[idx];
        atomicAdd(reinterpret_cast<unsigned long long*>(&histogram[custkey]), 1ULL);
    }
}
