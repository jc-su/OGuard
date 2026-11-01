#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to filter rows where supplycost equals min_supplycost (per-row comparison)
extern "C" __global__ void kernel(
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ min_supplycost_ptr,
    int64_t* __restrict__ keep_indices,
    int64_t* __restrict__ keep_count,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double cost = supplycost_ptr[idx];
        double min_c = min_supplycost_ptr[idx];
        bool keep = (cost == min_c);
        if (keep) {
            unsigned long long int* count_ptr = (unsigned long long int*)keep_count;
            unsigned long long int pos = atomicAdd(count_ptr, (unsigned long long int)1);
            keep_indices[pos] = idx;
        }
    }
}
