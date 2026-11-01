extern "C" __global__ void kernel(
    const int64_t* __restrict__ partkey_in,
    const double* __restrict__ supplycost,
    const double* __restrict__ availqty,
    int64_t* __restrict__ partkey_out,
    double* __restrict__ value_out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        partkey_out[idx] = partkey_in[idx];
        value_out[idx] = supplycost[idx] * availqty[idx];
    }
}
