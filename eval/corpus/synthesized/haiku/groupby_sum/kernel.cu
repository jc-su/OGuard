#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* group, const double* value, int64_t capacity,
    int32_t* table_keys, double* table_sums)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < gridDim.x * blockDim.x) {
        int32_t key = group[i];
        int64_t hash_val = key & (capacity - 1);

        while (table_keys[hash_val] != (int32_t)-1 && table_keys[hash_val] != key) {
            hash_val = (hash_val + 1) & (capacity - 1);
        }

        if (table_keys[hash_val] == (int32_t)-1) {
            table_keys[hash_val] = key;
        }
        atomicAdd(table_sums + hash_val, value[i]);
    }
}
