// hash_distinct: insert the distinct keys into an open-addressing table with linear
// probing from slot h(key) & (capacity - 1), h(x) = (x * 0x9E3779B97F4A7C15) >> 32 on
// the unsigned 64-bit value; table_rows[s] = smallest row with the key of slot s.
//
// One thread per row. A slot changes once, from -1 to a key, by compare-and-swap,
// so a key is never stored twice and never moves. A plain read of a slot can only
// be stale by showing -1; the CAS then returns the key that is there. Each thread
// lowers table_rows[s] with atomicMin, skipped when the slot already holds a row
// that is not larger (the stored row only decreases). The table holds at least
// twice as many slots as keys, so probing ends.

extern "C" __global__ void kernel(const long long* __restrict__ keys,
                                  long long* table_keys,
                                  int* table_rows,
                                  long long n, long long capacity)
{
    const unsigned long long mask = (unsigned long long)capacity - 1;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long key = keys[i];
        unsigned long long slot = (((unsigned long long)key * 0x9E3779B97F4A7C15ull) >> 32) & mask;
        while (true) {
            long long cur = table_keys[slot];
            if (cur == -1) {
                cur = (long long)atomicCAS((unsigned long long*)&table_keys[slot], ~0ull,
                                           (unsigned long long)key);
                if (cur == -1) cur = key;  // inserted here
            }
            if (cur == key) {
                if (table_rows[slot] > i) atomicMin(&table_rows[slot], (int)i);
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
}
