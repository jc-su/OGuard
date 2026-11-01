#include <stdint.h>

extern "C" __global__ void kernel(int64_t *build, int64_t *probe, int32_t *offsets, 
                                   int32_t *out_build, int32_t *out_probe, int64_t b, int64_t p) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p) {
        int32_t start_idx = offsets[i];
        int32_t end_idx = offsets[i + 1];
        
        for (int32_t k = start_idx; k < end_idx; k++) {
            int32_t idx_in_group = k - start_idx;
            int32_t j = start_idx + idx_in_group;
            
            out_build[k] = j;
            out_probe[k] = i;
        }
    }
}
