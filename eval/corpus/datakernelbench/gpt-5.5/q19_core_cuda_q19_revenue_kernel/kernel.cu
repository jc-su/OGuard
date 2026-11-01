#include <cstdint>

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
    __shared__ double block_sum[BLOCK_SIZE];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double my_sum = 0.0;

    if (idx < n) {
        double qty = l_quantity[idx];
        int32_t brand = p_brand_id[idx];
        int32_t cont = p_container_grp[idx];
        int32_t sz = p_size[idx];

        bool cond1 = (brand == brand12_id) &&
                     (cont == sm_grp) &&
                     (qty >= 1.0) &&
                     (qty <= 11.0) &&
                     (sz >= 1) &&
                     (sz <= 5);

        bool cond2 = (brand == brand23_id) &&
                     (cont == md_grp) &&
                     (qty >= 10.0) &&
                     (qty <= 20.0) &&
                     (sz >= 1) &&
                     (sz <= 10);

        bool cond3 = (brand == brand34_id) &&
                     (cont == lg_grp) &&
                     (qty >= 20.0) &&
                     (qty <= 30.0) &&
                     (sz >= 1) &&
                     (sz <= 15);

        if (cond1 || cond2 || cond3) {
            double price = l_extendedprice[idx];
            double disc = l_discount[idx];
            my_sum = price * (1.0 - disc);
        }
    }

    block_sum[threadIdx.x] = my_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}
