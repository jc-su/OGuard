#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ availqty_ptr,
    const int64_t* __restrict__ inverse_ptr,
    double* __restrict__ agg_ptr,
    int64_t n_elements,
    int64_t n_unique_keys
) {
    extern __shared__ double shared_vals[];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory for this block's portion of unique keys
    // Each block handles a subset of the aggregation space
    for (int i = tid; i < n_unique_keys; i += blockDim.x) {
        shared_vals[i] = 0.0;
    }
    __syncthreads();
    
    // Process elements: compute value and accumulate in shared memory by inverse index
    if (idx < n_elements) {
        double value = supplycost_ptr[idx] * availqty_ptr[idx];
        int64_t key_idx = inverse_ptr[idx];
        // Atomic add to shared memory position for this key
        atomicAdd(&shared_vals[key_idx], value);
    }
    __syncthreads();
    
    // Reduce shared memory to global: each thread handles a subset of keys
    for (int i = tid; i < n_unique_keys; i += blockDim.x) {
        if (shared_vals[i] > 0.0) {
            atomicAdd(&agg_ptr[i], shared_vals[i]);
        }
    }
}
