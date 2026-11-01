# bucketize (Triton)

- One row per thread (BLOCK=1024, 8 warps). Each row runs a binary search for
  the first bound greater than its value, which gives the count of bounds
  that are less than or equal to it (the same as `np.searchsorted(side="right")`).
- The search loop is a `while` on a scalar tensor, `tl.max(hi - lo) > 0`,
  because the step count depends on the runtime scalar `b` and the kernel has
  no constexpr for it. Rows past `n` start with `hi = 0` so they take no
  iterations; each iteration loads `bounds[mid]` under the mask `lo < hi`.
- Benchmark at n = 1000003, b = 1000 on the RTX A6000: 87 to 95 us across
  BLOCK in {512, 1024} and 4 or 8 warps.
