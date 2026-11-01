#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ extendedprice_ptr,
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
    int64_t numel
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (idx < numel) {
        int32_t brand = brand_id_ptr[idx];
        int32_t container = container_grp_ptr[idx];
        
        bool c1 = (brand == brand12_id) && (container == sm_grp);
        bool c2 = (brand == brand23_id) && (container == md_grp);
        bool c3 = (brand == brand34_id) && (container == lg_grp);
        
        if (c1 || c2 || c3) {
            double qty = quantity_ptr[idx];
            int32_t sz = size_ptr[idx];
            bool keep = false;
            
            if (c1 && qty >= 1.0 && qty <= 11.0 && sz >= 1 && sz <= 5) keep = true;
            else if (c2 && qty >= 10.0 && qty <= 20.0 && sz >= 1 && sz <= 10) keep = true;
            else if (c3 && qty >= 20.0 && qty <= 30.0 && sz >= 1 && sz <= 15) keep = true;
            
            if (keep) {
                double ep = extendedprice_ptr[idx];
                double disc = discount_ptr[idx];
                my_rev = ep * (1.0 - disc);
            }
        }
    }
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}
