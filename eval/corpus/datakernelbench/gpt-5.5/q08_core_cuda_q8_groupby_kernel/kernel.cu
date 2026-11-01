#include <cstdint>

#define BLOCK_SIZE 256
#define YEAR_BASE 1900
#define NUM_YEARS 201

extern "C" __global__ void kernel(
    const int32_t* __restrict__ o_year,
    const double* __restrict__ volume,
    const bool* __restrict__ is_brazil,
    double* __restrict__ sums,
    int64_t n_elements
) {
    extern __shared__ double shmem[];
    double* sh_total = shmem;
    double* sh_brazil = shmem + NUM_YEARS;

    int tid = threadIdx.x;

    for (int i = tid; i < NUM_YEARS; i += blockDim.x) {
        sh_total[i] = 0.0;
        sh_brazil[i] = 0.0;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    if (idx < n_elements) {
        int year_idx = static_cast<int>(o_year[idx]) - YEAR_BASE;
        if (year_idx >= 0 && year_idx < NUM_YEARS) {
            double v = volume[idx];
            atomicAdd(&sh_total[year_idx], v);
            if (is_brazil[idx]) {
                atomicAdd(&sh_brazil[year_idx], v);
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < NUM_YEARS; i += blockDim.x) {
        double t = sh_total[i];
        double b = sh_brazil[i];
        if (t != 0.0) {
            atomicAdd(&sums[i], t);
        }
        if (b != 0.0) {
            atomicAdd(&sums[NUM_YEARS + i], b);
        }
    }
}
