# days_to_year

Approach (Numba and CUDA): one thread per row, Hinnant's civil_from_days
integer arithmetic reduced to the year: shift the epoch to 0000-03-01, split
into 400-year eras (floor division, since dates before 1970 give negative
day counts) and the day of era, derive the year of era and the day of the
March-based year, and add one when that day is in January or February.
The Numba version uses Python floor division; the CUDA version emulates it
with the usual `(z >= 0 ? z : z - 146096) / 146097`.

Tests (n in 0, 1, 1000, 1000003): random days over years 1 to 9999 plus
the first and last day, Feb 28 and Mar 1 of years around the leap rules
(1, 4, 100, 400, 1600, 1900, 2000, 2100, 2400, 9999). The Python reference
is NumPy datetime64; the C++ reference tabulates year starts from the leap
rule and uses a binary search.

    CUDA_VISIBLE_DEVICES=1 python -P days_to_year/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
