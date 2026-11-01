# run_end_decode (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

Each program decodes 2048 rows [lo, lo + 2048). It finds the run holding row
lo, j0 = #{j : run_ends[j] <= lo}, with a 512-ary search over run_ends (three
steps for r = 10^8). At most 2048 runs overlap the tile, namely j0, j0 + 1, ...;
their ends that fall inside the tile mark where a new run starts, so a
histogram of (run end - lo) over 2048 bins, followed by an inclusive scan,
gives the run of every row. The values are then gathered from run_values.
The work per tile is the same for any run length distribution.

## Tests

n in {0, 1, 1000, 1000003} with one run, runs of one row, short runs (1 to 7
rows) and mixed runs (mostly 1 to 3 rows, 1% of 1000 to 5000 rows), compared
with `np.repeat`. The test also compiles the kernel with the declared `TYPES`.

## Performance (RTX A6000, n = 10^8)

670 to 709 GB/s (output plus both input arrays) for mean run lengths of 1, 4,
100 and 100 000 rows. A per-row binary search over run_ends was up to 3.5 times
slower for short runs.
