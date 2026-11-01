extern "C" __global__ void kernel(
    const int32_t* __restrict__ supp,
    const int32_t* __restrict__ cust,
    const int32_t* __restrict__ shipdate,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ revenue,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t sd = shipdate[idx];
    if (sd < date_lo || sd > date_hi) return;

    // Compute year from epoch days (civil_from_days algorithm)
    int64_t days = (int64_t)sd + 719468;
    int64_t era = (days >= 0 ? days : days - 146096) / 146097;
    int64_t doe = days - era * 146097;
    int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t y = yoe + era * 400;
    int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int64_t mp = (5 * doy + 2) / 153;
    int64_t m = (mp < 10) ? mp + 3 : mp - 9;
    int64_t year = (m <= 2) ? y + 1 : y;

    int year_offset = (int)(year - 1995);
    if (year_offset < 0 || year_offset > 1) return;

    int s = supp[idx];
    int c = cust[idx];
    int group = s * 4 + c * 2 + year_offset;
    double vol = price[idx] * (1.0 - discount[idx]);
    atomicAdd(&revenue[group], vol);
}
