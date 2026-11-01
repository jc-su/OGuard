#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ years,
    const double* __restrict__ volume,
    const uint8_t* __restrict__ is_brazil,
    double* __restrict__ sum_total,
    double* __restrict__ sum_brazil,
    const int32_t year_lo,
    const int32_t num_years,
    const int64_t N)
{
    // shared memory holds two double arrays (one per aggregation)
    extern __shared__ double shmem[];
    double* sh_total  = shmem;                     // size = num_years
    double* sh_brazil = shmem + num_years;         // size = num_years

    // initialise shared buffers
    for (int i = threadIdx.x; i < num_years; i += blockDim.x) {
        sh_total[i]  = 0.0;
        sh_brazil[i] = 0.0;
    }
    __syncthreads();

    // grid‑stride loop over rows
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        int32_t y = years[i];
        int off = y - year_lo;                     // map to 0 … num_years‑1
        if (off >= 0 && off < num_years) {
            double vol = volume[i];
            atomicAdd(&sh_total[off], vol);
            if (is_brazil[i])
                atomicAdd(&sh_brazil[off], vol);
        }
    }
    __syncthreads();

    // flush per‑block partials to global output (one atomic per year)
    for (int i = threadIdx.x; i < num_years; i += blockDim.x) {
        if (sh_total[i] != 0.0)
            atomicAdd(&sum_total[i], sh_total[i]);
        if (sh_brazil[i] != 0.0)
            atomicAdd(&sum_brazil[i], sh_brazil[i]);
    }
}
