// days_to_year: the proleptic Gregorian year of each day count since 1970-01-01.
//
// The civil-from-days algorithm (H. Hinnant) restricted to the year: shift the
// epoch to 0000-03-01 so that the leap day ends the year, split the day count
// into 400-year eras of 146097 days, and find the year of the era from the day
// of the era. A date in January or February belongs to the next civil year.
//
// For dates in the years 1 to 9999, the shifted day count z is in
// [306, 3652424], so all the arithmetic is unsigned 32-bit and every division
// is by a constant (a multiply-high). The kernel uses a grid-stride loop, so
// it is correct for any grid; the launch uses one thread per row.
//
// Launch: 256 threads per block, ceil(n / 256) blocks.
#include <cstdint>

namespace days_to_year {
constexpr int kBlock = 256;
}  // namespace days_to_year

extern "C" __global__ void __launch_bounds__(days_to_year::kBlock)
kernel(const int32_t* __restrict__ days, int32_t* __restrict__ year, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const uint32_t z = static_cast<uint32_t>(__ldg(days + i) + 719468);
    const uint32_t era = z / 146097u;
    const uint32_t doe = z - era * 146097u;                                    // [0, 146096]
    const uint32_t yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;  // [0, 399]
    const uint32_t doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);           // [0, 365]
    year[i] = static_cast<int32_t>(yoe + era * 400u + (doy >= 306u ? 1u : 0u));
  }
}
