#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* probe_keys, int64_t m,
    const int64_t* table_keys, const int32_t* table_rows, int64_t capacity,
    int32_t* out_build, int32_t* out_probe, int32_t* out_count, int64_t out_cap)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int64_t key = probe_keys[i];
        int64_t hash_val = key & (capacity - 1);

        while (table_keys[hash_val] != (int64_t)-1) {
            if (table_keys[hash_val] == key) {
                int32_t idx = atomicAdd(out_count, 1);
                if (idx >= 0 && idx < out_cap) {
                    out_build[idx] = table_rows[hash_val];
                    out_probe[idx] = (int32_t)i;
                }
                break;
            }
            hash_val = (hash_val + 1) & (capacity - 1);
        }
    }
}
