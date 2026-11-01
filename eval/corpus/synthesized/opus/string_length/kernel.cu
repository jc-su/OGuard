// Task 9, string_length: the number of UTF-8 code points of each string.
//
// The string column is in the Arrow layout: string i is the byte range
// chars[offsets[i], offsets[i + 1]). A code point starts at every byte that is
// not a continuation byte (0b10xxxxxx), so the kernel counts those bytes.
// One thread handles one string; the grid-stride loop accepts any grid size.
#include <cstdint>

extern "C" __global__ void kernel(const int32_t* __restrict__ offsets,
                                  const uint8_t* __restrict__ chars,
                                  int32_t* __restrict__ lengths,
                                  int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const int32_t end = offsets[i + 1];
    int32_t count = 0;
    for (int32_t j = offsets[i]; j < end; ++j) {
      count += (chars[j] & 0xC0) != 0x80;
    }
    lengths[i] = count;
  }
}
