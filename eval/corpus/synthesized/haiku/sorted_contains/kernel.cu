#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(int64_t *keys, int64_t *members, int8_t *match, int64_t n, int64_t s) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        
        int64_t left = 0, right = s;
        bool found = false;
        
        while (left < right) {
            int64_t mid = (left + right) / 2;
            if (members[mid] == key) {
                found = true;
                break;
            } else if (members[mid] < key) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        match[idx] = found ? 1 : 0;
    }
}
