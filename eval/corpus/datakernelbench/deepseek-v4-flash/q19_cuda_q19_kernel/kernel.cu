#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ brand_id_ptr,
    const int32_t* __restrict__ container_grp_ptr,
    const int32_t* __restrict__ size_ptr,
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
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;

    if (idx < n_elements) {
        int32_t bid = brand_id_ptr[idx];
        int32_t cg = container_grp_ptr[idx];
        double qty = quantity_ptr[idx];
        int32_t sz = size_ptr[idx];
        bool keep = false;

        if (bid == brand12_id) {
            keep = (cg == sm_grp) && (qty >= 1.0) && (qty <= 11.0) && (sz >= 1) && (sz <= 5);
        } else if (bid == brand23_id) {
            keep = (cg == md_grp) && (qty >= 10.0) && (qty <= 20.0) && (sz >= 1) && (sz <= 10);
        } else if (bid == brand34_id) {
            keep = (cg == lg_grp) && (qty >= 20.0) && (qty <= 30.0) && (sz >= 1) && (sz <= 15);
        }

        if (keep) {
            my_rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
        }
    }

    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}
