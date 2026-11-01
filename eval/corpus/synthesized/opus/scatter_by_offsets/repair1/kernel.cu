// Scatter by offsets: write the row id of each flagged row to its offset.
//
// offsets is the exclusive prefix sum of flags, so with flags of 0 or 1 the
// flagged rows land in out[0:total] in row order. This is the scatter step of
// stream compaction.
//
// total is the length of out. The kernel writes a row id only when its offset
// is in [0, total). For offsets that are the exclusive prefix sum of flags of
// 0 or 1, every flagged offset is in that range, so the check skips no write.
// The check keeps each store inside out for any value in offsets.

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
            const long long position = offsets[i];
            if (position >= 0 && position < total) {
                out[position] = static_cast<int>(i);
            }
        }
    }
}
