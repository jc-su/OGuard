// Kernel 2: Scan pre-filtered lineitems, probe the order map, and aggregate revenue
// directly into a per-customer array via atomicAdd.
extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ order_map,
    double* __restrict__ customer_revenue,
    int64_t max_orderkey,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t ok = l_orderkey[idx];
        if (ok >= 0 && ok <= max_orderkey) {
            int32_t ck = order_map[ok];
            if (ck >= 0) { // Valid match from orders
                double rev = l_extendedprice[idx] * (1.0 - l_discount[idx]);
                atomicAdd(&customer_revenue[ck], rev);
            }
        }
    }
}
