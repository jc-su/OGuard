#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ l_quantity_ptr,
    const double* __restrict__ l_extendedprice_ptr,
    const double* __restrict__ l_discount_ptr,
    const int32_t* __restrict__ p_brand_id_ptr,
    const int32_t* __restrict__ p_container_grp_ptr,
    const int32_t* __restrict__ p_size_ptr,
    const int32_t brand12_id,
    const int32_t brand23_id,
    const int32_t brand34_id,
    const int32_t sm_grp,
    const int32_t md_grp,
    const int32_t lg_grp,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double l_quantity = l_quantity_ptr[idx];
        bool cond1 = (p_brand_id_ptr[idx] == brand12_id) &&
                     (p_container_grp_ptr[idx] == sm_grp) &&
                     (l_quantity >= 1.0) && (l_quantity <= 11.0) &&
                     (p_size_ptr[idx] >= 1) && (p_size_ptr[idx] <= 5);
        bool cond2 = (p_brand_id_ptr[idx] == brand23_id) &&
                     (p_container_grp_ptr[idx] == md_grp) &&
                     (l_quantity >= 10.0) && (l_quantity <= 20.0) &&
                     (p_size_ptr[idx] >= 1) && (p_size_ptr[idx] <= 10);
        bool cond3 = (p_brand_id_ptr[idx] == brand34_id) &&
                     (p_container_grp_ptr[idx] == lg_grp) &&
                     (l_quantity >= 20.0) && (l_quantity <= 30.0) &&
                     (p_size_ptr[idx] >= 1) && (p_size_ptr[idx] <= 15);
        double my_rev = 0.0;
        if (cond1 || cond2 || cond3) {
            double price = l_extendedprice_ptr[idx];
            double discount = l_discount_ptr[idx];
            my_rev = price * (1.0 - discount);
        }
        block_revenue[threadIdx.x] = my_rev;
    } else {
        block_revenue[threadIdx.x] = 0.0;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}
