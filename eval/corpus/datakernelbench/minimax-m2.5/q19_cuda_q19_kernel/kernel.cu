#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ p_brand_id,
    const int32_t* __restrict__ p_container_grp,
    const int32_t* __restrict__ p_size,
    double* __restrict__ out,
    int64_t n,
    int32_t brand12_id,
    int32_t brand23_id,
    int32_t brand34_id,
    int32_t sm_grp,
    int32_t md_grp,
    int32_t lg_grp,
    double q1_lo, double q1_hi,
    double q2_lo, double q2_hi,
    double q3_lo, double q3_hi,
    int32_t s1_lo, int32_t s1_hi,
    int32_t s2_lo, int32_t s2_hi,
    int32_t s3_lo, int32_t s3_hi
) {
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0.0;
    if (idx < n) {
        double qty = l_quantity[idx];
        double price = l_extendedprice[idx];
        double disc = l_discount[idx];
        int32_t brand = p_brand_id[idx];
        int32_t cont = p_container_grp[idx];
        int32_t sz = p_size[idx];
        bool keep = false;
        if (brand == brand12_id && cont == sm_grp && qty >= q1_lo && qty <= q1_hi && sz >= s1_lo && sz <= s1_hi) keep = true;
        else if (brand == brand23_id && cont == md_grp && qty >= q2_lo && qty <= q2_hi && sz >= s2_lo && sz <= s2_hi) keep = true;
        else if (brand == brand34_id && cont == lg_grp && qty >= q3_lo && qty <= q3_hi && sz >= s3_lo && sz <= s3_hi) keep = true;
        if (keep) my_sum = price * (1.0 - disc);
    }
    block_sum[threadIdx.x] = my_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, block_sum[0]);
}
