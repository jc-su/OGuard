#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ double atomicMin(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMax(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

extern "C" __global__ void kernel(double *values, double *result, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = blockDim.x;
    int tid = threadIdx.x;
    
    extern __shared__ char shared_mem[];
    double *shared_min = (double *)shared_mem;
    double *shared_max = (double *)shared_mem + block_size;
    
    double local_min = INFINITY;
    double local_max = -INFINITY;
    
    if (idx < n) {
        double val = values[idx];
        if (val < local_min) local_min = val;
        if (val > local_max) local_max = val;
    }
    
    // Ensure tid is in bounds for shared memory write
    if (tid < block_size) {
        shared_min[tid] = local_min;
        shared_max[tid] = local_max;
    }
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s /= 2) {
        // Explicit bounds check before shared memory access
        if (tid < s) {
            int access_idx = tid + s;
            // Ensure both indices are in bounds
            if (access_idx < block_size) {
                if (shared_min[access_idx] < shared_min[tid])
                    shared_min[tid] = shared_min[access_idx];
                if (shared_max[access_idx] > shared_max[tid])
                    shared_max[tid] = shared_max[access_idx];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMin(result, shared_min[0]);
        atomicMax(result + 1, shared_max[0]);
    }
}
