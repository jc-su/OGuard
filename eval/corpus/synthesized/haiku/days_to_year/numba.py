from numba import cuda, types

def kernel(days, year, n):
    """Convert days since 1970-01-01 to year."""
    idx = cuda.grid(1)
    if idx < n:
        day_count = days[idx]

        y = 1970
        remaining = day_count

        if remaining >= 0:
            while True:
                if y % 400 == 0:
                    is_leap_year = True
                elif y % 100 == 0:
                    is_leap_year = False
                elif y % 4 == 0:
                    is_leap_year = True
                else:
                    is_leap_year = False
                
                days_in_year = 366 if is_leap_year else 365
                if remaining < days_in_year:
                    break
                remaining -= days_in_year
                y += 1
        else:
            y -= 1
            remaining = -remaining
            while True:
                if y % 400 == 0:
                    is_leap_year = True
                elif y % 100 == 0:
                    is_leap_year = False
                elif y % 4 == 0:
                    is_leap_year = True
                else:
                    is_leap_year = False
                
                days_in_year = 366 if is_leap_year else 365
                if remaining <= days_in_year:
                    break
                remaining -= days_in_year
                y -= 1

        year[idx] = y

TYPES = (types.int32[:], types.int32[:], types.int64)

def run(days_np, year_np, n):
    """Launch kernel on device arrays."""
    days = cuda.to_device(days_np)
    year = cuda.to_device(year_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](days, year, n)

    return year.copy_to_host()
