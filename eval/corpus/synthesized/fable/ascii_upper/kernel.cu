// ascii_upper: copy the column's bytes to out with ASCII a-z replaced by A-Z.
// The work is one item per byte; the byte count is offsets[n], so the grid
// strides over it and any grid size of at least one block is correct.
extern "C" __global__ void kernel(const int* offsets,
                                  const unsigned char* chars,
                                  unsigned char* out,
                                  long long n) {
    long long total = offsets[n];
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < total; j += stride) {
        unsigned char c = chars[j];
        if (c >= 'a' && c <= 'z') c -= 32;
        out[j] = c;
    }
}
