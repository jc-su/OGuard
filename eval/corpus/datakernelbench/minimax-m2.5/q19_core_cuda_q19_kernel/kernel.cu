#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ p_brand_id,
    const int32_t* __restrict__ p_container_grp,
    const int32_t* __restrict__ p_size,
    int32_t brand12_id,
    int32_t brand23_id,
    int32_t brand34_id,
    int32_t sm_grp,
    int32_t md_grp,
    int32_t lg_grp,
    double* __restrict__ out,
    int64_t n
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    if (idx < n) {
        double qty = l_quantity[idx];
        double price = l_extendedprice[idx];
        double disc = l_discount[idx];
        int32_t brand = p_brand_id[idx];
        int32_t cont = p_container_grp[idx];
        int32_t sz = p_size[idx];
        bool keep = false;
        // cond1: brand12, SM container, qty [1,11], size [1,5]
        if (brand == brand12_id && cont == sm_grp && qty >= 1.0 && qty <= 11.0 && sz >= 1 && sz <= 5)
            keep = true;
        // cond2: brand23, MED container, qty [10,20], size [1,10]
        if (!keep && brand == brand23_id && cont == md_grp && qty >= 10.0 && qty <= 20.0 && sz >= 1 && sz <= 10)
            keep = true;
        // cond3: brand34, LG container, qty [20,30], size [1,15]
        if (!keep && brand == brand34_id && cont == lg_grp && qty >= 20.0 && qty <= 30.0 && sz >= 1 && sz <= 15)
            keep = true;
        if (keep)
            my_rev = price * (1.0 - disc);
    }
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, block_revenue[0]);
}
