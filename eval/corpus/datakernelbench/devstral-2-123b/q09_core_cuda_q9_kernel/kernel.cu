extern "C" __global__ void kernel(
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const double* __restrict__ ps_supplycost,
    const double* __restrict__ l_quantity,
    double* __restrict__ amount_out,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        amount_out[idx] = l_extendedprice[idx] * (1.0 - l_discount[idx]) - ps_supplycost[idx] * l_quantity[idx];
    }
}
