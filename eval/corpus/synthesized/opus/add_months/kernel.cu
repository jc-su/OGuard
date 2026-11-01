// add_months: out[i] = days[i] plus months[i] months, keeping the day of the month or
// clamping it to the last day of the target month.
//
// Dates convert with Howard Hinnant's civil_from_days / days_from_civil algorithms
// on 400-year eras. Every date is in the years 1 to 9999, so all the shifted day
// and year counts are nonnegative and the divisions need no floor correction.
// The month arithmetic counts months from year 0: target = y * 12 + (m - 1) + months.

__device__ __forceinline__ int days_in_month(int y, int m)
{
    if (m == 2) return ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 29 : 28;
    return 30 + ((m + (m >> 3)) & 1);
}

extern "C" __global__ void kernel(const int* __restrict__ days,
                                  const int* __restrict__ months,
                                  int* __restrict__ out,
                                  long long n)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // civil_from_days
        const int z = days[i] + 719468;
        const int era = z / 146097;
        const int doe = z - era * 146097;
        const int yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        const int doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const int mp = (5 * doy + 2) / 153;
        const int d = doy - (153 * mp + 2) / 5 + 1;
        const int m = mp < 10 ? mp + 3 : mp - 9;
        const int y = yoe + era * 400 + (m <= 2);

        // Target month and day.
        const int target = y * 12 + (m - 1) + months[i];
        const int ty = target / 12;
        const int tm = target - ty * 12 + 1;
        const int dim = days_in_month(ty, tm);
        const int td = d < dim ? d : dim;

        // days_from_civil
        const int yy = ty - (tm <= 2);
        const int era2 = yy / 400;
        const int yoe2 = yy - era2 * 400;
        const int doy2 = (153 * (tm > 2 ? tm - 3 : tm + 9) + 2) / 5 + td - 1;
        const int doe2 = yoe2 * 365 + yoe2 / 4 - yoe2 / 100 + doy2;
        out[i] = era2 * 146097 + doe2 - 719468;
    }
}
