// days_to_year: convert a day count since 1970-01-01 (proleptic
// Gregorian) into its calendar year, for dates in years 1-9999.
//
// Uses Howard Hinnant's `civil_from_days` algorithm
// (http://howardhinnant.github.io/date_algorithms.html), which computes
// the year in O(1) integer arithmetic -- no loops, no libc date
// functions. For years >= 1 its "proleptic Gregorian, year 0 = 1 BC"
// numbering coincides with the usual 1-9999 numbering, so no offset is
// needed. One thread per row.
extern "C" __global__ void kernel(const int *__restrict__ days,
                                   int *__restrict__ year, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long long z = (long long)days[i] + 719468;
    long long era = (z >= 0 ? z : z - 146096) / 146097;
    long long doe = z - era * 146097;                             // [0, 146096]
    long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    long long y = yoe + era * 400;
    long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);       // [0, 365]
    long long mp = (5 * doy + 2) / 153;                            // [0, 11]
    long long m = mp + (mp < 10 ? 3 : -9);                         // [1, 12]
    year[i] = (int)(y + (m <= 2 ? 1 : 0));
  }
}
