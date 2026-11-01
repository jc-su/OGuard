// Scatter by offsets: write the row id of each flagged row to its offset.
//
// offsets is the exclusive prefix sum of flags, so with flags of 0 or 1 the
// flagged rows land in out[0:total] in row order. This is the scatter step of
// stream compaction. total is the length of out; the kernel does not read it.

extern "C" __global__ void kernel(const signed char* __restrict__ flags,
                                  const int* __restrict__ offsets,
                                  int* __restrict__ out,
                                  long long n,
                                  long long total)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        if (flags[i] != 0) {
            out[offsets[i]] = static_cast<int>(i);
        }
    }
}
