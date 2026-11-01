from numba import cuda, types

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def days_in_month(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    return 0

def kernel(days, months, out, n):
    i = cuda.grid(1)
    if i < n:
        d = days[i]
        m = months[i]
        
        year = 1970
        month = 1
        day = 1
        
        total_days = d
        while total_days > 365:
            days_this_year = 366 if is_leap_year(year) else 365
            if total_days > days_this_year:
                total_days -= days_this_year
                year += 1
            else:
                break
        
        month = 1
        day = total_days
        while day > days_in_month(year, month):
            day -= days_in_month(year, month)
            month += 1
        
        target_month = month + m
        target_year = year
        while target_month > 12:
            target_month -= 12
            target_year += 1
        while target_month < 1:
            target_month += 12
            target_year -= 1
        
        max_day = days_in_month(target_year, target_month)
        if day > max_day:
            day = max_day
        
        result_days = 0
        for y in range(1970, target_year):
            result_days += 366 if is_leap_year(y) else 365
        
        for m in range(1, target_month):
            result_days += days_in_month(target_year, m)
        
        result_days += day - 1
        out[i] = result_days

TYPES = (types.int32[:], types.int32[:], types.int32[:], types.int64)

def run(days, months, out, n):
    cuda.jit(kernel)[(n + 255) // 256, 256](days, months, out, n)
