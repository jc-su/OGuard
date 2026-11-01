// bitmap_slice: out bit i = source bit offset + i, for i < n; bits n and above of the
// last output byte are 0.
//
// Output byte k takes bits [offset + 8k, offset + 8k + 8): the high part of source
// byte offset / 8 + k and, when offset % 8 != 0, the low part of the next byte.
// The next byte is read only when it exists: source has (offset + n + 7) / 8 bytes.

extern "C" __global__ void kernel(const unsigned char* __restrict__ source,
                                  unsigned char* __restrict__ out,
                                  long long n, long long offset)
{
    const long long out_bytes = (n + 7) >> 3;
    const long long source_bytes = (offset + n + 7) >> 3;
    const long long first = offset >> 3;
    const int s = (int)(offset & 7);
    const int tail = (int)(n & 7);
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < out_bytes; k += stride) {
        const long long b = first + k;
        unsigned int v = (unsigned int)source[b] >> s;
        if (s != 0 && b + 1 < source_bytes) v |= (unsigned int)source[b + 1] << (8 - s);
        if (k == out_bytes - 1 && tail != 0) v &= (1u << tail) - 1u;
        out[k] = (unsigned char)v;
    }
}
