#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ nation_id_ptr,
    double* __restrict__ out_ptr,
    int64_t num_nations,
    int64_t n_elements
) {
    // Per-block shared memory for reduction
    __shared__ double block_revenue[BLOCK_SIZE];
    // Also store per-thread nation_id for the active thread
    __shared__ int64_t block_nation[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    int64_t my_nid = 0;

    if (idx < n_elements) {
        my_nid = nation_id_ptr[idx];
        double price = price_ptr[idx];
        double disc = discount_ptr[idx];
        my_rev = price * (1.0 - disc);
    }

    // Store into shared memory
    block_revenue[threadIdx.x] = my_rev;
    block_nation[threadIdx.x] = my_nid;
    __syncthreads();

    // Block reduction for sum (but we can't just sum because different nation_ids)
    // For Q5, each block may have multiple nations; we need to atomically add to correct bucket
    // Better: per-thread atomicAdd to global out array based on nation_id
    // Since groups are small (~25 nations), collisions are rare; atomicAdd is fine here
    // but we still use shared mem reduce per nation? No, too complex.
    // Simpler: each thread does atomicAdd to its nation's bucket directly.
    // To reduce atomics, we do a block-level gather of distinct nations first? Overkill.
    // For small num_nations (<=25), per-thread atomicAdd is acceptable since blocks >> nations.
    // Let's just let each surviving thread do atomicAdd.

    // Reset shared memory for potential block-level reduction per nation (complex)
    // Easiest fusion: each thread directly atomics to out_ptr[nation_id]
    // Since num_nations <= 25, contention is low.

    __syncthreads(); // ensure all shared stores visible

    // Use shared memory to handle at most 1024 threads per block, each adding to its nation
    // To reduce global atomics from 1024 to at most num_nations per block:
    // For each possible nation_id (0..num_nations-1), reduce in shared memory then single atomic

    if (threadIdx.x < num_nations) {
        int64_t target_nation = (int64_t)threadIdx.x;
        double sum_for_nation = 0.0;

        // Check each thread in block for this nation
        // We'll do a parallel reduction across threads for each nation
        // Since num_nations is small (< 100 typically), this is fine
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (block_nation[i] == target_nation) {
                sum_for_nation += block_revenue[i];
            }
        }

        if (sum_for_nation != 0.0) {
            atomicAdd(&out_ptr[target_nation], sum_for_nation);
        }
    }
}
