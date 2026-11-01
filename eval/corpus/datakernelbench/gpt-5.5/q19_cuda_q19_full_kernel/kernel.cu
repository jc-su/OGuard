#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ brand_ptr,
    const int32_t* __restrict__ container_ptr,
    const int32_t* __restrict__ size_ptr,
    const int32_t brand12_id,
    const int32_t brand23_id,
    const int32_t brand34_id,
    const int32_t sm_grp,
    const int32_t md_grp,
    const int32_t lg_grp,
    double* __restrict__ out_ptr,
    const int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    double my_rev = 0.0;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_elements;
         idx += stride) {
        const double qty = quantity_ptr[idx];
        const int32_t brand = brand_ptr[idx];
        const int32_t cont = container_ptr[idx];
        const int32_t psz = size_ptr[idx];

        const bool cond1 =
            (brand == brand12_id) &&
            (cont == sm_grp) &&
            (qty >= 1.0) &&
            (qty <= 11.0) &&
            (psz >= 1) &&
            (psz <= 5);

        const bool cond2 =
            (brand == brand23_id) &&
            (cont == md_grp) &&
            (qty >= 10.0) &&
            (qty <= 20.0) &&
            (psz >= 1) &&
            (psz <= 10);

        const bool cond3 =
            (brand == brand34_id) &&
            (cont == lg_grp) &&
            (qty >= 20.0) &&
            (qty <= 30.0) &&
            (psz >= 1) &&
            (psz <= 15);

        if (cond1 || cond2 || cond3) {
            const double price = price_ptr[idx];
            const double disc = discount_ptr[idx];
            my_rev += price * (1.0 - disc);
        }
    }

    block_revenue[tid] = my_rev;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_revenue[tid] += block_revenue[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}
