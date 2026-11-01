#include <stdint.h>

extern "C" __global__ void kernel(int64_t *keys, int64_t *table_keys, int32_t *table_rows, int64_t n, int64_t capacity) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t key = keys[i];
        uint64_t h = ((uint64_t)key * 0x9E3779B97F4A7C15) >> 32;
        int64_t slot = h & (capacity - 1);
        
        while (table_keys[slot] != -1 && table_keys[slot] != key) {
            slot = (slot + 1) & (capacity - 1);
        }
        
        if (table_keys[slot] == -1) {
            table_keys[slot] = key;
            table_rows[slot] = i;
        } else if (table_rows[slot] > i) {
            atomicMin(&table_rows[slot], (int32_t)i);
        }
    }
}
