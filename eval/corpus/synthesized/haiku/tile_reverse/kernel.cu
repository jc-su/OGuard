extern "C" __global__ void kernel(int* values, int* out, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long tile_id = idx / 1024;
        long long pos_in_tile = idx % 1024;
        long long tile_start = tile_id * 1024;
        long long tile_end = (tile_start + 1024 < n) ? tile_start + 1024 : n;
        long long tile_len = tile_end - tile_start;
        
        long long rev_pos = tile_len - 1 - pos_in_tile;
        out[idx] = values[tile_start + rev_pos];
    }
}
