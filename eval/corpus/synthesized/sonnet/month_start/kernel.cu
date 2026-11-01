// Civil-from-days (proleptic Gregorian, March-based years). The month start is
// the day number minus (day of month - 1). Inputs are >= -719162, so z is positive.
extern "C" __global__ void kernel(const int* days, int* out, long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int day = days[i];
        int z = day + 719468;
        int era = z / 146097;
        int doe = z - era * 146097;
        int yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        int doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        int mp = (5 * doy + 2) / 153;
        int dom = doy - (153 * mp + 2) / 5 + 1;
        out[i] = day - (dom - 1);
    }
}
