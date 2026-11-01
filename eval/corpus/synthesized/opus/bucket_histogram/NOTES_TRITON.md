# bucket_histogram (Triton)

- Each row finds its bucket with a branchless binary search over `bounds`. The search counts the bounds that are <= the value, so ties go to the upper bucket and +inf and -inf are handled.
- When b + 1 <= LOCAL = 256, each program builds a histogram of its 1024 rows with `tl.histogram` and adds only the nonzero bins with atomics. This avoids per-row atomics on a few hot counters. For larger b, each row adds 1 to its bucket with an atomic, because the counters are spread out.
- Tests: `python -P test_triton.py` with n in {0, 1, 1000, 1000003} and b in {0, 1, 10, 255, 256, 5000}. b = 255 and b = 256 cover both paths.
