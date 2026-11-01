// string_equal: match[i] = 1 when string i of column a equals string i of
// column b byte for byte, and 0 otherwise. One thread per row.
extern "C" __global__ void kernel(const int* a_offsets,
                                  const unsigned char* a_chars,
                                  const int* b_offsets,
                                  const unsigned char* b_chars,
                                  signed char* match,
                                  long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int a0 = a_offsets[i];
        int b0 = b_offsets[i];
        int length = a_offsets[i + 1] - a0;
        bool equal = length == b_offsets[i + 1] - b0;
        for (int j = 0; equal && j < length; ++j) {
            equal = a_chars[a0 + j] == b_chars[b0 + j];
        }
        match[i] = equal ? 1 : 0;
    }
}
