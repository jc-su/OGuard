// One thread per row: civil date from days (Hinnant's algorithm, all
// intermediate values nonnegative for years 1 to 9999), month arithmetic,
// day clamped to the target month, and back to days.
extern "C" __global__ void kernel(const int* days, const int* months, int* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long z = (long long)days[i] + 719468;
    long long era = z / 146097;
    long long doe = z - era * 146097;
    long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    long long mp = (5 * doy + 2) / 153;
    long long d = doy - (153 * mp + 2) / 5 + 1;
    long long m = mp < 10 ? mp + 3 : mp - 9;
    long long y = yoe + era * 400 + (m <= 2 ? 1 : 0);
    long long total = y * 12 + (m - 1) + months[i];
    long long ty = total / 12;
    long long tm = total - ty * 12 + 1;
    long long dim;
    if (tm == 2) {
        bool leap = (ty % 4 == 0 && ty % 100 != 0) || ty % 400 == 0;
        dim = leap ? 29 : 28;
    } else {
        dim = 31 - ((tm - 1) % 7) % 2;
    }
    if (d > dim) d = dim;
    long long yy = tm <= 2 ? ty - 1 : ty;
    long long era2 = yy / 400;
    long long yoe2 = yy - era2 * 400;
    long long mp2 = tm > 2 ? tm - 3 : tm + 9;
    long long doy2 = (153 * mp2 + 2) / 5 + d - 1;
    long long doe2 = yoe2 * 365 + yoe2 / 4 - yoe2 / 100 + doy2;
    out[i] = (int)(era2 * 146097 + doe2 - 719468);
}
