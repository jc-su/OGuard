// string_hash: out[i] = 32-bit FNV-1a hash of the bytes of string i.
//
// FNV-1a is sequential within a string, so each thread hashes one string.
// Neighboring threads hash neighboring strings, so their byte loads fall in the
// same cache lines.

extern "C" __global__ void kernel(const int* __restrict__ offsets,
                                  const unsigned char* __restrict__ chars,
                                  unsigned int* __restrict__ out,
                                  long long n)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long begin = offsets[i];
        const long long end = offsets[i + 1];
        unsigned int h = 2166136261u;
        for (long long p = begin; p < end; ++p) h = (h ^ chars[p]) * 16777619u;
        out[i] = h;
    }
}
