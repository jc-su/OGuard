#include <cuda_runtime.h>
#include <stdint.h>
#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ ps_partkey,
    const double* __restrict__ ps_supplycost,
    const double* __restrict__ ps_availqty,
    int64_t* __restrict__ unique_keys_out,
    double* __restrict__ grouped_values_out,
    int64_t* __restrict__ key_count_out,
    double* __restrict__ total_rev_ptr,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    int64_t my_key = 0;
    
    if (idx < n) {
        my_key = ps_partkey[idx];
        my_rev = ps_supplycost[idx] * ps_availqty[idx];
    }

    // Block reduce for total
    __shared__ double block_totals[BLOCK_SIZE];
    block_totals[threadIdx.x] = my_rev;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_totals[threadIdx.x] += block_totals[threadIdx.x + s];
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
        atomicAdd(total_rev_ptr, block_totals[0]);

    // Use shared memory hash table for grouping
    __shared__ double bucket_vals[2048];
    __shared__ int64_t bucket_keys[2048];
    __shared__ int bucket_used[2048];
    
    if (threadIdx.x < 2048) {
        bucket_vals[threadIdx.x] = 0.0;
        bucket_keys[threadIdx.x] = -1;
        bucket_used[threadIdx.x] = 0;
    }
    __syncthreads();
    
    if (idx < n) {
        unsigned int hash = (unsigned int)(my_key % 2048);
        
        while (atomicCAS(&bucket_used[hash], 0, 1) != 0) {
            if (bucket_keys[hash] == my_key) {
                atomicAdd(&bucket_vals[hash], my_rev);
                goto done;
            }
            hash = (hash + 1) % 2048;
        }
        bucket_keys[hash] = my_key;
        bucket_vals[hash] = my_rev;
    }
    done:;
    
    __syncthreads();
    
    if (threadIdx.x < 2048 && bucket_used[threadIdx.x]) {
        int64_t pos = atomicAdd((unsigned long long*)key_count_out, 1ULL);
        unique_keys_out[pos] = bucket_keys[threadIdx.x];
        grouped_values_out[pos] = bucket_vals[threadIdx.x];
    }
}
