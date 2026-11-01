#include <cuda.h>
#include <cuda_runtime.h>

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
        int32_t brand_id = brand_id_ptr[idx];
        int32_t cont_grp = container_grp_ptr[idx];
        int32_t p_size = size_ptr[idx];
        double l_qty = quantity_ptr[idx];
        double l_price = price_ptr[idx];
        double l_disc = discount_ptr[idx];
        
        bool cond1 = (brand_id == brand12_id) 
                  && (cont_grp == sm_grp)
                  && (l_qty >= 1.0) && (l_qty <= 11.0)
                  && (p_size >= 1) && (p_size <= 5);
        
        bool cond2 = (brand_id == brand23_id)
                  && (cont_grp == md_grp)
                  && (l_qty >= 10.0) && (l_qty <= 20.0)
                  && (p_size >= 1) && (p_size <= 10);
        
        bool cond3 = (brand_id == brand34_id)
                  && (cont_grp == lg_grp)
                  && (l_qty >= 20.0) && (l_qty <= 30.0)
                  && (p_size >= 1) && (p_size <= 15);
        
        if (cond1 || cond2 || cond3) {
            my_rev = l_price * (1.0 - l_disc);
        }
    }
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }
    
    // One atomicAdd per block
    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}
