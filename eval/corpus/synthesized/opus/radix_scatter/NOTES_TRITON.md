# radix_scatter (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

Each program takes 1024 rows and ranks them locally: it sorts the packed pairs
(digit, local row) with `tl.sort`, so the rows of one digit become contiguous
and in row order, builds the tile histogram with `tl.histogram`, and reserves
each digit's output range with one vector atomic add of the histogram on
`cursors` (one atomic per digit present instead of one per row). The row at
sorted position q with digit d goes to cursor_before[d] + q - tile_start[d],
so the stores of a digit are contiguous. Rows keep their order inside a digit
within a tile.

## Tests

n in {0, 1, 1000, 1000003}; uniform keys, one constant key and four distinct
keys; shift in {0, 8, 13, 24}. The output must hold every row id once, grouped
by digit in cursor order, and `cursors` must end at start + count. The test
also compiles the kernel with the declared `TYPES`.

## Performance (RTX A6000, n = 5 * 10^7)

Uniform digits 1.44 ms, one digit 1.23 ms, skewed digits 1.32 ms; one atomic
per row takes 14.6 to 28.7 ms. The local sort is the largest cost; larger
tiles coalesce better but sort slower (2048 rows: 1.8 ms).
