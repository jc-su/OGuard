#include <stdint.h>

extern "C" __global__ void kernel(int64_t *values, int32_t *run_ends, int64_t *run_values, int32_t *count, int64_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && n > 0) {
        int32_t run_count = 0;
        int64_t j = 0;
        
        while (j < n) {
            int64_t current_val = values[j];
            int64_t run_start = j;
            
            while (j < n && values[j] == current_val) {
                j++;
            }
            
            run_ends[run_count] = j;
            run_values[run_count] = current_val;
            run_count++;
        }
        
        count[0] = run_count;
    }
}
