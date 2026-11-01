#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int64_t *keys, int64_t *members, int8_t *match, int64_t n, int64_t s) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        
        int64_t left = 0, right = s - 1;
        int8_t found = 0;
        
        while (left <= right) {
            int64_t mid = (left + right) / 2;
            // Explicit bounds check immediately before array access
            if (mid >= 0 && mid < s) {
                int64_t members_mid = members[mid];
                if (members_mid == key) {
                    found = 1;
                    break;
                } else if (members_mid < key) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                // Exit loop if mid is out of bounds
                break;
            }
        }
        
        match[idx] = found;
    }
}
