# days_to_year

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003} (random days over the full valid range) plus fixed
points (1970-01-01, the day before/after, and the extreme days of years
1 and 9999), all pass.

Approach: Howard Hinnant's `civil_from_days` algorithm
(http://howardhinnant.github.io/date_algorithms.html) -- closed-form
integer arithmetic, no loops, no libc date/time calls. One thread per
row. Verified independently against Python's `datetime` module (in
test_numba.py) and against Hinnant's inverse `days_from_civil` formula
plus a binary search over the year (in test.cu) -- both are a different
code path from the kernel's formula, so they're a genuine cross-check,
not a restatement of the same derivation. All valid inputs (years
1-9999) give a non-negative `z` internally, so the algorithm's negative
branch is never exercised by in-spec data, but it is still implemented
and was validated separately for completeness.
