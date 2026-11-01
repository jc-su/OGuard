// One thread per output byte: eight consecutive rows of the column.
extern "C" __global__ void kernel(const long long* values, const unsigned char* valid,
                                  unsigned char* out, long long n, long long lo, long long hi) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nbytes = (n + 7) / 8;
    if (b >= nbytes) return;
    long long base = b * 8;
    int rows = (int)((n - base < 8) ? (n - base) : 8);
    unsigned int vbits = valid[b];
    unsigned int acc = 0;
    for (int k = 0; k < rows; ++k) {
        if ((vbits >> k) & 1u) {
            long long x = values[base + k];
            if (x >= lo && x <= hi) acc |= 1u << k;
        }
    }
    out[b] = (unsigned char)acc;
}
