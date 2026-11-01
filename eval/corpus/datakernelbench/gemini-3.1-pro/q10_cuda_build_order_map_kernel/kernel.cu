// Kernel 1: Build a hash map (direct array) of valid orders to their customer keys.
// Filters orders by date range on the fly.
extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_orderkey,
    const int64_t* __restrict__ o_custkey,
    const int32_t* __restrict__ o_orderdate,
    int32_t* __restrict__ order_map,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t date = o_orderdate[idx];
        if (date >= date_lo && date < date_hi) {
            order_map[o_orderkey[idx]] = (int32_t)o_custkey[idx];
        }
    }
}
