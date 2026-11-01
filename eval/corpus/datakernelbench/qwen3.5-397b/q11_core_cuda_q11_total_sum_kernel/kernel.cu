#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ availqty_ptr,
    double* __restrict__ total_ptr,
    int64_t n_elements
) {
    extern __shared__ double shared_sum[];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Each thread computes partial sum
    double my_sum = 0.0;
    if (idx < n_elements) {
        my_sum = supplycost_ptr[idx] * availqty_ptr[idx];
    }
    shared_sum[tid] = my_sum;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes block sum to global
    if (tid == 0) {
        atomicAdd(total_ptr, shared_sum[0]);
    }
}
