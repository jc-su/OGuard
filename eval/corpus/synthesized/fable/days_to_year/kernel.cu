// days_to_year: year of a date given as days since 1970-01-01.
//
// One thread per row, branch-free integer arithmetic (Howard Hinnant's
// civil_from_days): shift the epoch to 0000-03-01, split into 400-year eras
// and the day of era, derive the year of era and the day of the March-based
// year, and add one when that day falls in January or February.

extern "C" __global__ void kernel(const int* __restrict__ days,
                                  int* __restrict__ year,
                                  long long n)
{
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int z = days[i] + 719468;                          // days since 0000-03-01
    const int era = (z >= 0 ? z : z - 146096) / 146097;      // floor division
    const unsigned doe = (unsigned)(z - era * 146097);       // day of era, 0 to 146096
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // 0 to 399
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                // 0 to 365
    year[i] = (int)yoe + era * 400 + (doy >= 306 ? 1 : 0);
}
