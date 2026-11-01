#include <stdint.h>

__device__ static int is_leap_year(int32_t year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

__device__ static int days_in_month(int32_t year, int32_t month) {
    if (month == 2) return is_leap_year(year) ? 29 : 28;
    if (month == 4 || month == 6 || month == 9 || month == 11) return 30;
    return 31;
}

extern "C" __global__ void kernel(int32_t *days, int32_t *months, int32_t *out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t d = days[i];
        int32_t m = months[i];
        
        int32_t year = 1970;
        int32_t month = 1;
        int32_t day = 1;
        
        int32_t total_days = d;
        while (total_days > 365) {
            int32_t days_this_year = is_leap_year(year) ? 366 : 365;
            if (total_days > days_this_year) {
                total_days -= days_this_year;
                year++;
            } else {
                break;
            }
        }
        
        month = 1;
        day = total_days;
        while (day > days_in_month(year, month)) {
            day -= days_in_month(year, month);
            month++;
        }
        
        int32_t target_month = month + m;
        int32_t target_year = year;
        while (target_month > 12) {
            target_month -= 12;
            target_year++;
        }
        while (target_month < 1) {
            target_month += 12;
            target_year--;
        }
        
        int32_t max_day = days_in_month(target_year, target_month);
        if (day > max_day) day = max_day;
        
        int32_t result_days = 0;
        for (int32_t y = 1970; y < target_year; y++) {
            result_days += is_leap_year(y) ? 366 : 365;
        }
        
        for (int32_t mo = 1; mo < target_month; mo++) {
            result_days += days_in_month(target_year, mo);
        }
        
        result_days += day - 1;
        out[i] = result_days;
    }
}
