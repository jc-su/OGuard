#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Kernel for TPC-H Q8 Aggregation
// Computes sum(volume) and sum(brazil_volume) grouped by year (1995, 1996)
// Index 0 -> 1995, Index 1 -> 1996
extern "C" __global__ void kernel(
    const int32_t* __restrict__ years,
    const double* __restrict__ prices,
    const double* __restrict__ discounts,
    const bool* __restrict__ is_brazil,
    double* __restrict__ out_total,
    double* __restrict__ out_brazil,
    int64_t n_elements
) {
    // Shared memory layout (4 * BLOCK_SIZE doubles):
    // [0..BLOCK_SIZE-1] = total_1995
    // [BLOCK_SIZE..2*BLOCK_SIZE-1] = total_1996
    // [2*BLOCK_SIZE..3*BLOCK_SIZE-1] = brazil_1995
    // [3*BLOCK_SIZE..4*BLOCK_SIZE-1] = brazil_1996
    extern __shared__ double sdata[];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double reg_total_1995 = 0.0;
    double reg_total_1996 = 0.0;
    double reg_brazil_1995 = 0.0;
    double reg_brazil_1996 = 0.0;

    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n_elements; i += stride) {
        double vol = prices[i] * (1.0 - discounts[i]);
        int32_t y = years[i];
        bool is_br = is_brazil[i];
        
        if (y == 1995) {
            reg_total_1995 += vol;
            if (is_br) reg_brazil_1995 += vol;
        } else if (y == 1996) {
            reg_total_1996 += vol;
            if (is_br) reg_brazil_1996 += vol;
        }
    }

    // Write to shared memory with proper offsets to avoid overlap
    sdata[tid] = reg_total_1995;
    sdata[BLOCK_SIZE + tid] = reg_total_1996;
    sdata[2 * BLOCK_SIZE + tid] = reg_brazil_1995;
    sdata[3 * BLOCK_SIZE + tid] = reg_brazil_1996;
    
    __syncthreads();

    // Reduction for all 4 accumulators
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[BLOCK_SIZE + tid] += sdata[BLOCK_SIZE + tid + s];
            sdata[2 * BLOCK_SIZE + tid] += sdata[2 * BLOCK_SIZE + tid + s];
            sdata[3 * BLOCK_SIZE + tid] += sdata[3 * BLOCK_SIZE + tid + s];
        }
        __syncthreads();
    }

    // Thread 0 atomics to global memory
    if (tid == 0) {
        atomicAdd(&out_total[0], sdata[0]);              // 1995 total
        atomicAdd(&out_total[1], sdata[BLOCK_SIZE]);     // 1996 total
        atomicAdd(&out_brazil[0], sdata[2 * BLOCK_SIZE]); // 1995 brazil
        atomicAdd(&out_brazil[1], sdata[3 * BLOCK_SIZE]); // 1996 brazil
    }
}
