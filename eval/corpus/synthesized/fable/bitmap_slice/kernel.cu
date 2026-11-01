// One thread per output byte, assembled from the two source bytes that hold
// its bits.
extern "C" __global__ void kernel(const unsigned char* source, unsigned char* out, long long n, long long offset) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nbytes = (n + 7) / 8;
    if (b >= nbytes) return;
    long long src_bytes = (offset + n + 7) / 8;
    long long q = (offset >> 3) + b;
    int sh = (int)(offset & 7);
    unsigned int acc = (unsigned int)source[q] >> sh;
    if (sh != 0 && q + 1 < src_bytes) acc |= (unsigned int)source[q + 1] << (8 - sh);
    long long rows = n - b * 8;
    if (rows < 8) acc &= (1u << rows) - 1u;
    out[b] = (unsigned char)acc;
}
