# days_to_year (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

Civil-from-days (H. Hinnant) in int32 arithmetic with divisions by constants
only: shift the day number to count from 0000-03-01, split it into 400-year
eras, years of era and day of year; a day of year >= 306 (January or February)
belongs to the next calendar year. For years 1 to 9999 the shifted day number
is positive, so truncating division equals floor division.

## Tests

n in {0, 1, 1000, 1000003} with random dates in the years 1 to 9999, and every
day from 0001-01-01 to 9999-12-31 (3.65 million days), compared with NumPy
`datetime64` conversion. The test also compiles the kernel with the declared
`TYPES`.

## Performance (RTX A6000, n = 10^8)

1.17 ms, 681 GB/s, the same as a device copy of the same bytes.
