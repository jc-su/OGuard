#include <stdint.h>
#include <cuda_runtime.h>

__device__ bool is_leap(int32_t year) {
    if (year % 400 == 0) return true;
    if (year % 100 == 0) return false;
    if (year % 4 == 0) return true;
    return false;
}

extern "C" __global__ void kernel(int32_t *days, int32_t *year, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t day_count = days[idx];
        
        int32_t y = 1970;
        int32_t remaining = day_count;
        
        if (remaining >= 0) {
            while (true) {
                int32_t days_in_year = is_leap(y) ? 366 : 365;
                if (remaining < days_in_year) break;
                remaining -= days_in_year;
                y++;
            }
        } else {
            y--;
            remaining = -remaining;
            while (true) {
                int32_t days_in_year = is_leap(y) ? 366 : 365;
                if (remaining <= days_in_year) break;
                remaining -= days_in_year;
                y--;
            }
        }
        
        year[idx] = y;
    }
}
