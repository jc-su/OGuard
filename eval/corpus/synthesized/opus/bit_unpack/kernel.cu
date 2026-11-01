// bit_unpack: out[i] = the bits-wide code at bit position i * bits of the LSB-first
// stream in packed (Parquet bit-packed dictionary codes).
//
// Code i starts at bit p = i * bits (up to 2^35, so 64-bit), in word p / 32 at shift
// p % 32. The two words p / 32 and p / 32 + 1 form a 64-bit window that holds the
// whole code; the second word is read only when the code reaches into it, which
// also keeps the read inside the w words.

extern "C" __global__ void kernel(const unsigned int* __restrict__ packed,
                                  unsigned int* __restrict__ out,
                                  long long n, long long w, long long bits)
{
    const unsigned long long mask = bits >= 32 ? 0xffffffffull : ((1ull << bits) - 1);
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long p = i * bits;
        const long long word = p >> 5;
        const int shift = (int)(p & 31);
        unsigned long long window = packed[word];
        if (shift + bits > 32) window |= (unsigned long long)packed[word + 1] << 32;
        out[i] = (unsigned int)((window >> shift) & mask);
    }
}
