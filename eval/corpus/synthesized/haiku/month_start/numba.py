from numba import cuda
import numpy as np


def kernel(days, out, n):
    """Find the first day of the month for each day."""
    idx = cuda.grid(1)
    if idx < n:
        day_number = days[idx]

        # Epoch: 1970-01-01 = day 0
        year = 1970
        month = 1
        current_day = 0

        # Fast forward to the year and month
        while current_day + 365 < day_number:
            # Check if leap year
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            days_this_year = 366 if is_leap else 365
            if current_day + days_this_year <= day_number:
                current_day += days_this_year
                year += 1
            else:
                break

        # Find the month
        while True:
            # Days in current month
            if month in [1, 3, 5, 7, 8, 10, 12]:
                days_this_month = 31
            elif month in [4, 6, 9, 11]:
                days_this_month = 30
            elif month == 2:
                is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
                days_this_month = 29 if is_leap else 28
            else:
                days_this_month = 0

            if current_day + days_this_month > day_number:
                out[idx] = current_day
                return

            current_day += days_this_month
            month += 1
            if month > 12:
                month = 1
                year += 1


TYPES = (
    cuda.int32[:],
    cuda.int32[:],
    cuda.int64,
)


def run(days, out, n, grid_size=None):
    """Launch the month_start kernel."""
    if grid_size is None:
        grid_size = max(1, (n + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](days, out, n)
