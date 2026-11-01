extern "C" __global__ void kernel(
    const int32_t* __restrict__ o_year,
    const double* __restrict__ volume,
    const bool* __restrict__ is_brazil,
    int64_t n_elements,
    double* __restrict__ sum_brazil,
    double* __restrict__ sum_total,
    int32_t min_year,
    int32_t num_years
) {
    // Each block uses shared memory to accumulate for all years
    // Use shared memory array indexed by year
    extern __shared__ double smem[];
    double* sh_brazil = smem;
    double* sh_total = smem + num_years;
    
    // Initialize shared memory for this block
    int tid = threadIdx.x;
    if (tid < num_years) {
        sh_brazil[tid] = 0.0;
        sh_total[tid] = 0.0;
    }
    __syncthreads();
    
    // Each thread processes multiple elements
    int64_t idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    
    for (int64_t i = idx; i < n_elements; i += stride) {
        int32_t year_idx = o_year[i] - min_year;
        if (year_idx >= 0 && year_idx < num_years) {
            double vol = volume[i];
            if (is_brazil[i]) {
                atomicAdd(&sh_brazil[year_idx], vol);
            }
            atomicAdd(&sh_total[year_idx], vol);
        }
    }
    __syncthreads();
    
    // Write block results to global
    if (tid < num_years) {
        atomicAdd(&sum_brazil[tid], sh_brazil[tid]);
        atomicAdd(&sum_total[tid], sh_total[tid]);
    }
}
