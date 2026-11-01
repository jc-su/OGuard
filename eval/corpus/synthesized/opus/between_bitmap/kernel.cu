// between_bitmap: out bit i = valid(i) && lo <= values[i] <= hi, as an Arrow bitmap.
//
// Each warp evaluates 32 consecutive rows, assembles their bits with a ballot,
// and lanes 0..3 store the four output bytes. A warp's 32 rows start at a
// multiple of 32, so every output byte is written by exactly one lane and the
// bits at positions n and above are 0.

extern "C" __global__ void kernel(const long long* __restrict__ values,
                                  const unsigned char* __restrict__ valid,
                                  unsigned char* __restrict__ out,
                                  long long n, long long lo, long long hi)
{
    const int lane = threadIdx.x & 31;
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long long num_warps = ((long long)gridDim.x * blockDim.x) >> 5;
    const long long out_bytes = (n + 7) >> 3;

    for (long long base = warp * 32; base < n; base += num_warps * 32) {
        const long long i = base + lane;
        bool bit = false;
        if (i < n) {
            const long long v = values[i];
            bit = ((valid[i >> 3] >> (i & 7)) & 1) && lo <= v && v <= hi;
        }
        const unsigned int word = __ballot_sync(0xffffffffu, bit);
        const long long byte = (base >> 3) + lane;
        if (lane < 4 && byte < out_bytes) {
            out[byte] = (unsigned char)(word >> (8 * lane));
        }
    }
}
