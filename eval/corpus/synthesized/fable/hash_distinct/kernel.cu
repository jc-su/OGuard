// One thread per key: linear probing from the hash slot. A slot that holds -1
// is claimed with a compare-and-swap; a slot never changes again once it holds
// a key, so a plain read that returns a key is final. Whichever thread finds
// the key's slot lowers table_rows[slot] with an atomic min.
extern "C" __global__ void kernel(const long long* keys, long long* table_keys, int* table_rows,
                                  long long n, long long capacity) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long key = keys[i];
    unsigned long long h = ((unsigned long long)key * 0x9E3779B97F4A7C15ull) >> 32;
    long long mask = capacity - 1;
    long long slot = (long long)h & mask;
    while (true) {
        long long cur = table_keys[slot];
        if (cur == -1)
            cur = (long long)atomicCAS((unsigned long long*)&table_keys[slot], (unsigned long long)-1ll, (unsigned long long)key);
        if (cur == -1 || cur == key) {
            atomicMin(&table_rows[slot], (int)i);
            break;
        }
        slot = (slot + 1) & mask;
    }
}
