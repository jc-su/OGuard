// Task 10, starts_with: match[i] = 1 when string i starts with the byte pattern prefix.
//
// The string column is in the Arrow layout: string i is the byte range
// chars[offsets[i], offsets[i + 1]). The comparison is bytewise, so for a valid
// UTF-8 pattern it is also a code point prefix test. An empty pattern (k == 0)
// matches every string. One thread handles one string; the grid-stride loop
// accepts any grid size.
#include <cstdint>

extern "C" __global__ void kernel(const int32_t* __restrict__ offsets,
                                  const uint8_t* __restrict__ chars,
                                  const uint8_t* __restrict__ prefix,
                                  int8_t* __restrict__ match,
                                  int64_t n,
                                  int64_t k) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const int64_t begin = offsets[i];
    int8_t result = 0;
    if (offsets[i + 1] - begin >= k) {
      result = 1;
      for (int64_t j = 0; j < k; ++j) {
        if (chars[begin + j] != prefix[j]) {
          result = 0;
          break;
        }
      }
    }
    match[i] = result;
  }
}
