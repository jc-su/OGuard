#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int*    __restrict__ p_brand_id,
    const int*    __restrict__ p_container_grp,
    const int*    __restrict__ p_size,
    const int     brand12_id,
    const int     brand23_id,
    const int     brand34_id,
    const int     sm_grp,
    const int     md_grp,
    const int     lg_grp,
    double*       __restrict__ out,
    const int64_t n) {

    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    double my_rev = 0.0;

    if (idx < n) {
        // Load once per thread (coalesced)
        double qty   = l_quantity[idx];
        double price = l_extendedprice[idx];
        double disc  = l_discount[idx];
        int    brand = p_brand_id[idx];
        int    cont  = p_container_grp[idx];
        int    size  = p_size[idx];

        bool keep = false;

        // condition 1
        if (brand == brand12_id && cont == sm_grp &&
            qty >= 1.0 && qty <= 11.0 &&
            size >= 1 && size <= 5) {
            keep = true;
        }
        // condition 2
        if (!keep && brand == brand23_id && cont == md_grp &&
            qty >= 10.0 && qty <= 20.0 &&
            size >= 1 && size <= 10) {
            keep = true;
        }
        // condition 3
        if (!keep && brand == brand34_id && cont == lg_grp &&
            qty >= 20.0 && qty <= 30.0 &&
            size >= 1 && size <= 15) {
            keep = true;
        }

        if (keep) {
            my_rev = price * (1.0 - disc);
        }
    }

    // block‑wise reduction
    block_sum[threadIdx.x] = my_rev;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // one atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}
