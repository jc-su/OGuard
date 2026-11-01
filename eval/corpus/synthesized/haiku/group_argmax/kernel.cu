#include <stdint.h>

extern "C" __global__ void kernel(int32_t *group, int64_t *value, int32_t *best, int64_t n, int64_t g) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t grp = group[i];
        int64_t val = value[i];
        
        int32_t old_best = best[grp];
        int64_t old_best_val = (old_best == -1) ? INT64_MIN : value[old_best];
        
        if (val > old_best_val) {
            atomicMin(&best[grp], (int32_t)i);
        }
    }
}
