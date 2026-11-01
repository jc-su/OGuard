# days_to_year: Triton

Status: complete. `test_triton.py` passes for n in {0, 1, 1000, 1000003} with uniform dates over the years 1 to 9999 and dates near the epoch, plus every Jan 1, Dec 31, Feb 28, Feb 29 or Mar 1 boundary of the years 1 to 9999 and the limits 0001-01-01 and 9999-12-31, against NumPy datetime64.

Approach: one row per lane, 1024 rows per program. Hinnant's civil_from_days arithmetic: shift the epoch to 0000-03-01 (z = days + 719468, positive for years >= 1, so truncating division is exact), split into 400-year eras and years within the era, and add one when the March-based month is January or February. Divisions by constants compile to multiply-high sequences.

Time at n = 1000003: 14 us.
