#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ part_keys,
    int32_t* __restrict__ part_map,
    int64_t n_parts,
    int64_t map_size
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_parts;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t key = part_keys[i];
        if (key >= 0 && key < map_size) {
            part_map[key] = (int32_t)i;
        }
    }
}
