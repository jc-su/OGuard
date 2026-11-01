// starts_with: match[i] = 1 when string i begins with the k-byte pattern prefix.
//
// One thread per string: a string shorter than k never matches; otherwise the
// first k bytes are compared with the pattern and the loop stops at the first
// mismatch. The pattern is tiny and uniform across threads, so its loads are
// served from L1 as broadcasts.

#include <cstdint>

extern "C" __global__ void kernel(const int* __restrict__ offsets,
                                  const uint8_t* __restrict__ chars,
                                  const uint8_t* __restrict__ prefix,
                                  int8_t* __restrict__ match,
                                  long long n,
                                  long long k) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int begin = offsets[i];
        bool ok = (long long)(offsets[i + 1] - begin) >= k;
        for (long long j = 0; ok && j < k; ++j) ok = chars[begin + j] == prefix[j];
        match[i] = ok ? 1 : 0;
    }
}
