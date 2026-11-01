// lag: out[i] = values[i - k] when i - k >= first[i], the first row of row i's
// partition, and NaN otherwise. first[i] >= 0, so the read index is >= 0.
#include <cstdint>

extern "C" __global__ void kernel(const double* __restrict__ values,
                                  const int32_t* __restrict__ first,
                                  double* __restrict__ out,
                                  int64_t n,
                                  int64_t k)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const int64_t j = i - k;
        // Quiet NaN with the bit pattern 0x7ff8000000000000, as NumPy and Arrow write it.
        out[i] = j >= first[i] ? values[j] : __longlong_as_double(0x7ff8000000000000LL);
    }
}
