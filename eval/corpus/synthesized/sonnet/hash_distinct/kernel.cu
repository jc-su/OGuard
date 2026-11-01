#include <stdint.h>

// One thread inserts one key. Slots only change from the empty marker -1 to a key, so
// every thread that probes the same key reaches the same slot and a key occupies one slot.
// table_rows[slot] takes the minimum row over all rows of that key.
extern "C" __global__ void kernel(const int64_t* __restrict__ keys, int64_t* table_keys,
                                  int32_t* table_rows, int64_t n, int64_t capacity) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;

    const int64_t key = keys[i];
    const uint64_t mask = (uint64_t)capacity - 1;
    uint64_t slot = (((uint64_t)key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
    while (true) {
        long long old = (long long)atomicCAS((unsigned long long*)&table_keys[slot],
                                             (unsigned long long)-1LL, (unsigned long long)key);
        if (old == -1LL || old == key) {
            atomicMin(&table_rows[slot], (int32_t)i);
            return;
        }
        slot = (slot + 1) & mask;
    }
}
