// ascii_upper: copy a string column's bytes chars[offsets[0], offsets[n]) to
// the same positions of out, with each ASCII letter a-z replaced by A-Z.
//
// Each thread converts one aligned 8-byte word per grid-stride step. chars and
// out must be 8-byte aligned, as Arrow requires of its buffers. The words at
// the two ends of the range that the range covers only in part are converted
// byte by byte, so the kernel never touches a byte outside the range.
#include <cstdint>

// Uppercase the ASCII lowercase letters of the eight bytes of x at once. For a
// byte b with low seven bits h, h + 0x1F has bit 7 set when h >= 'a', and
// h + 0x05 has bit 7 set when h > 'z'. Bytes >= 0x80 are not letters. The
// sums stay below 0x100, so no carry crosses into the next byte.
__device__ __forceinline__ uint64_t upper8(uint64_t x)
{
    const uint64_t low7 = x & 0x7F7F7F7F7F7F7F7FULL;
    const uint64_t at_least_a = low7 + 0x1F1F1F1F1F1F1F1FULL;
    const uint64_t above_z = low7 + 0x0505050505050505ULL;
    const uint64_t lower = at_least_a & ~above_z & ~x & 0x8080808080808080ULL;
    return x ^ (lower >> 2);  // Clear bit 5 (0x20) of each lowercase letter.
}

__device__ __forceinline__ uint8_t upper1(uint8_t c)
{
    return (c >= 'a' && c <= 'z') ? static_cast<uint8_t>(c - ('a' - 'A')) : c;
}

extern "C" __global__ void kernel(const int32_t* __restrict__ offsets,
                                  const uint8_t* __restrict__ chars,
                                  uint8_t* __restrict__ out,
                                  int64_t n)
{
    const int64_t begin = offsets[0];
    const int64_t end = offsets[n];
    const int64_t first_word = begin >> 3;
    const int64_t end_word = (end + 7) >> 3;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const uint64_t* __restrict__ chars_words = reinterpret_cast<const uint64_t*>(chars);
    uint64_t* __restrict__ out_words = reinterpret_cast<uint64_t*>(out);

    for (int64_t w = first_word + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         w < end_word; w += stride) {
        const int64_t lo = w << 3;
        const int64_t hi = lo + 8;
        if (lo >= begin && hi <= end) {
            out_words[w] = upper8(chars_words[w]);
        } else {
            const int64_t stop = hi < end ? hi : end;
            for (int64_t p = lo > begin ? lo : begin; p < stop; ++p) {
                out[p] = upper1(chars[p]);
            }
        }
    }
}
