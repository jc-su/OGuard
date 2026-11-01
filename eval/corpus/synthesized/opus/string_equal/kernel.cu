// string_equal: match[i] = 1 when string i of column a equals string i of
// column b byte for byte, and 0 otherwise. One thread compares one row.
#include <cstdint>

extern "C" __global__ void kernel(const int32_t* __restrict__ a_offsets,
                                  const uint8_t* __restrict__ a_chars,
                                  const int32_t* __restrict__ b_offsets,
                                  const uint8_t* __restrict__ b_chars,
                                  int8_t* __restrict__ match,
                                  int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const int64_t a_begin = a_offsets[i];
        const int64_t b_begin = b_offsets[i];
        const int64_t length = a_offsets[i + 1] - a_begin;
        bool equal = b_offsets[i + 1] - b_begin == length;
        for (int64_t k = 0; equal && k < length; ++k) {
            equal = a_chars[a_begin + k] == b_chars[b_begin + k];
        }
        match[i] = equal ? 1 : 0;
    }
}
