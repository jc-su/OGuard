// gather_nullable: out[i] = col[idx[i]], and NaN where idx[i] is -1
// (the unmatched rows of a left outer join).
#include <cstdint>

extern "C" __global__ void kernel(const double* __restrict__ col,
                                  const int32_t* __restrict__ idx,
                                  double* __restrict__ out,
                                  int64_t m,
                                  int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < m) {
        const int32_t j = idx[i];
        // Quiet NaN with the bit pattern 0x7ff8000000000000, as NumPy and Arrow write it.
        out[i] = j >= 0 ? col[j] : __longlong_as_double(0x7ff8000000000000LL);
    }
}
