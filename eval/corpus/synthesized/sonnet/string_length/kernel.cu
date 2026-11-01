// GPU kernel for string_length.
//
// Counts the UTF-8 code points of each string i, delimited by
// offsets[i]:offsets[i+1] in chars. A byte starts a code point unless
// it is a continuation byte (its top two bits are 10), so counting
// non-continuation bytes counts code points.

extern "C" __global__ void kernel(const int *offsets, const unsigned char *chars,
                                   int *lengths, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int start = offsets[i];
    int end = offsets[i + 1];
    int count = 0;
    for (int j = start; j < end; ++j) {
        if ((chars[j] & 0xC0) != 0x80) count++;
    }
    lengths[i] = count;
}
