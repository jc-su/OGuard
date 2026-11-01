#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipdate_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ price_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    const int tid = threadIdx.x;
    double thread_sum = 0.0;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_elements;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int32_t shipdate = shipdate_ptr[idx];

        if ((shipdate >= date_lo) && (shipdate < date_hi)) {
            const double discount = discount_ptr[idx];

            if ((discount >= disc_lo) && (discount <= disc_hi)) {
                const double quantity = quantity_ptr[idx];

                if (quantity < qty_hi) {
                    thread_sum += price_ptr[idx] * discount;
                }
            }
        }
    }

    block_revenue[tid] = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_revenue[tid] += block_revenue[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}
