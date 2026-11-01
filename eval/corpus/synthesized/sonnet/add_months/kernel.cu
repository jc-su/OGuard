#include <stdint.h>

// Civil date arithmetic after Howard Hinnant. z counts days from 0000-03-01 and
// is nonnegative for the years 1 to 9999, so all divisions are of nonnegative values.
extern "C" __global__ void kernel(const int32_t* __restrict__ days,
                                  const int32_t* __restrict__ months,
                                  int32_t* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;

    int64_t z = (int64_t)days[i] + 719468;
    int64_t era = z / 146097;
    int64_t doe = z - era * 146097;
    int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t y = yoe + era * 400;
    int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int64_t mp = (5 * doy + 2) / 153;
    int64_t d = doy - (153 * mp + 2) / 5 + 1;
    int64_t m = mp < 10 ? mp + 3 : mp - 9;
    y += (m <= 2);

    int64_t total = y * 12 + (m - 1) + (int64_t)months[i];
    int64_t ty = total / 12;
    int64_t tm = total % 12 + 1;

    int64_t dim;
    if (tm == 2) {
        dim = ((ty % 4 == 0 && ty % 100 != 0) || ty % 400 == 0) ? 29 : 28;
    } else if (tm == 4 || tm == 6 || tm == 9 || tm == 11) {
        dim = 30;
    } else {
        dim = 31;
    }
    if (d > dim) d = dim;

    int64_t y2 = ty - (tm <= 2);
    int64_t era2 = y2 / 400;
    int64_t yoe2 = y2 - era2 * 400;
    int64_t doy2 = (153 * (tm > 2 ? tm - 3 : tm + 9) + 2) / 5 + d - 1;
    int64_t doe2 = yoe2 * 365 + yoe2 / 4 - yoe2 / 100 + doy2;
    out[i] = (int32_t)(era2 * 146097 + doe2 - 719468);
}
