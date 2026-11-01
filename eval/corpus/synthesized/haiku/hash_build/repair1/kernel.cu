#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* keys, int64_t n, int64_t capacity,
    int64_t* table_keys, int32_t* table_rows)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t key = keys[i];
        int64_t hash_val = key & (capacity - 1);

        while (table_keys[hash_val] != (int64_t)-1 && table_keys[hash_val] != key) {
            hash_val = (hash_val + 1) & (capacity - 1);
        }

        if (table_keys[hash_val] == (int64_t)-1) {
            table_keys[hash_val] = key;
        }
        table_rows[hash_val] = (int32_t)i;
    }
}
