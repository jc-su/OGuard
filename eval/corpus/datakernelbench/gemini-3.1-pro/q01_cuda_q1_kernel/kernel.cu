extern "C" __global__ void kernel(
    const int32_t* __restrict__ rf_ptr,
    const int32_t* __restrict__ ls_ptr,
    const double* __restrict__ qty_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    const double* __restrict__ tax_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t shipdate_hi,
    int32_t num_ls,
    int32_t num_groups,
    int32_t num_sets,
    double* __restrict__ sum_qty,
    double* __restrict__ sum_base_price,
    double* __restrict__ sum_disc_price,
    double* __restrict__ sum_charge,
    double* __restrict__ sum_disc,
    int64_t* __restrict__ count_order,
    int64_t n_elements
) {
    int warp_id = threadIdx.x / 32;
    int set_id = warp_id % num_sets;
    
    // Shared memory allocated dynamically
    extern __shared__ double smem[];
    int row_size = num_groups * 6;
    double* my_smem = smem + set_id * row_size;
    
    // Initialize shared memory for this block
    for (int i = threadIdx.x; i < num_sets * row_size; i += blockDim.x) {
        smem[i] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (shipdate_ptr[idx] <= shipdate_hi) {
            int group = rf_ptr[idx] * num_ls + ls_ptr[idx];
            double qty = qty_ptr[idx];
            double price = price_ptr[idx];
            double disc = disc_ptr[idx];
            double tax = tax_ptr[idx];
            
            atomicAdd(&my_smem[group * 6 + 0], qty);
            atomicAdd(&my_smem[group * 6 + 1], price);
            atomicAdd(&my_smem[group * 6 + 2], price * (1.0 - disc));
            atomicAdd(&my_smem[group * 6 + 3], price * (1.0 - disc) * (1.0 + tax));
            atomicAdd(&my_smem[group * 6 + 4], disc);
            
            // Re-use the 6th double as an unsigned long long counter
            unsigned long long* w_count = (unsigned long long*)&my_smem[group * 6 + 5];
            atomicAdd(w_count, 1ULL);
        }
    }
    __syncthreads();

    // Reduce local accumulators to global memory
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        double t_qty = 0, t_price = 0, t_disc_price = 0, t_charge = 0, t_disc = 0;
        unsigned long long t_count = 0;
        
        for (int s = 0; s < num_sets; ++s) {
            double* s_smem = smem + s * row_size;
            t_qty += s_smem[g * 6 + 0];
            t_price += s_smem[g * 6 + 1];
            t_disc_price += s_smem[g * 6 + 2];
            t_charge += s_smem[g * 6 + 3];
            t_disc += s_smem[g * 6 + 4];
            t_count += *(unsigned long long*)&s_smem[g * 6 + 5];
        }
        
        if (t_count > 0) {
            atomicAdd(&sum_qty[g], t_qty);
            atomicAdd(&sum_base_price[g], t_price);
            atomicAdd(&sum_disc_price[g], t_disc_price);
            atomicAdd(&sum_charge[g], t_charge);
            atomicAdd(&sum_disc[g], t_disc);
            atomicAdd((unsigned long long*)&count_order[g], t_count);
        }
    }
}
