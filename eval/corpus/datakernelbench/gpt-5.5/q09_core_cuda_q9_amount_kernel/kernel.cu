#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const double* __restrict__ ps_supplycost,
    const double* __restrict__ l_quantity,
    double* __restrict__ out,
    int64_t n_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double discounted_price = l_extendedprice[idx] * (1.0 - l_discount[idx]);
        double supply_cost = ps_supplycost[idx] * l_quantity[idx];
        out[idx] = discounted_price - supply_cost;
    }
}
