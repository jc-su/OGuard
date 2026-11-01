#include <cuda_runtime.h>

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
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    double my_rev = 0.0;

    if (idx < n_elements) {
        int32_t brand = p_brand_id[idx];
        int32_t container = p_container_grp[idx];
        double qty = l_quantity[idx];
        int32_t size = p_size[idx];

        bool cond1 = (brand == brand12_id) && (container == sm_grp) && (qty >= 1.0) && (qty <= 11.0) && (size >= 1) && (size <= 5);
        bool cond2 = (brand == brand23_id) && (container == md_grp) && (qty >= 10.0) && (qty <= 20.0) && (size >= 1) && (size <= 10);
        bool cond3 = (brand == brand34_id) && (container == lg_grp) && (qty >= 20.0) && (qty <= 30.0) && (size >= 1) && (size <= 15);

        if (cond1 || cond2 || cond3) {
            my_rev = l_extendedprice[idx] * (1.0 - l_discount[idx]);
        }
    }

    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();

    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}
