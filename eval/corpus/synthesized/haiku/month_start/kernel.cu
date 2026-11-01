#include <cstdint>

__device__ bool is_leap_year_device(int32_t year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

__device__ int32_t days_in_month_device(int32_t year, int32_t month) {
    if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12) {
        return 31;
    } else if (month == 4 || month == 6 || month == 9 || month == 11) {
        return 30;
    } else if (month == 2) {
        return is_leap_year_device(year) ? 29 : 28;
    }
    return 0;
}

extern "C" __global__ void kernel(
    const int32_t* days,
    int32_t* out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t day_number = days[idx];

        int32_t year = 1970;
        int32_t month = 1;
        int32_t current_day = 0;

        // Fast forward to the year and month
        while (current_day + 365 < day_number) {
            int32_t days_this_year = is_leap_year_device(year) ? 366 : 365;
            if (current_day + days_this_year <= day_number) {
                current_day += days_this_year;
                year++;
            } else {
                break;
            }
        }

        // Find the month
        while (true) {
            int32_t days_this_month = days_in_month_device(year, month);
            if (current_day + days_this_month > day_number) {
                out[idx] = current_day;
                return;
            }
            current_day += days_this_month;
            month++;
            if (month > 12) {
                month = 1;
                year++;
            }
        }
    }
}
