#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipdate_days_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ extendedprice_ptr,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double discount_lo,
    double discount_hi,
    double quantity_hi,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double my_revenue = 0.0;

    if (idx < n_elements) {
        const int32_t shipdate_days = shipdate_days_ptr[idx];

        if ((shipdate_days >= shipdate_lo_days) && (shipdate_days < shipdate_hi_days)) {
            const double discount = discount_ptr[idx];

            if ((discount >= discount_lo) && (discount <= discount_hi)) {
                const double quantity = quantity_ptr[idx];

                if (quantity < quantity_hi) {
                    const double extendedprice = extendedprice_ptr[idx];
                    my_revenue = extendedprice * discount;
                }
            }
        }
    }

    block_revenue[threadIdx.x] = my_revenue;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}
