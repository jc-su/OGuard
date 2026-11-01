extern "C" __global__ void kernel(
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int64_t* __restrict__ nation_id_ptr,
    double* __restrict__ out_revenue,
    int64_t num_nations,
    int64_t num_rows
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rows) {
        int64_t nation_id = nation_id_ptr[idx];
        if (nation_id >= 0 && nation_id < num_nations) {
            double revenue = l_extendedprice[idx] * (1.0 - l_discount[idx]);
            atomicAdd(&out_revenue[nation_id], revenue);
        }
    }
}
