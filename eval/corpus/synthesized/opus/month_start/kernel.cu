// Month start: the day number of the first day of the month that contains each day.
#include <cstdint>

extern "C" __global__ void kernel(const int32_t* days, int32_t* out, int64_t n) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        // Civil date from day number (H. Hinnant), with years starting on March 1 so
        // that the leap day is the last day of the year.
        const int32_t day = days[i];
        const int32_t z = day + 719468;
        const int32_t era = (z >= 0 ? z : z - 146096) / 146097;
        const uint32_t doe = static_cast<uint32_t>(z - era * 146097);
        const uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        const uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const uint32_t mp = (5 * doy + 2) / 153;
        // doy - (153 * mp + 2) / 5 is the zero-based day of the month.
        out[i] = day - static_cast<int32_t>(doy - (153 * mp + 2) / 5);
    }
}
