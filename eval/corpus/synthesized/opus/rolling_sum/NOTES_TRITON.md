# rolling_sum (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

Van Herk / Gil-Werman: the rows are cut into blocks of `window` rows, and
out[i] = suffix_sum[i - window + 1] + prefix_sum[i] with both sums inside a
block. Each output is a sum of at most `window` inputs and nothing is
subtracted, so the error is that of summing the window directly (a prefix sum
difference would lose the small windows next to large values).

Float64 additions bound this kernel on the A6000 (two float64 units per SM),
so each program holds its 2048-row region as a [8, 256] tile whose columns are
8 consecutive rows in one thread: the segmented scans run sequentially down
the columns and then across the 256 column totals. The compiled kernel has 62
float64 additions per thread (SASS count) for 8 rows (about 8 per row); the same scheme
with Triton's scans over a flat 4096-row tile has 452 per thread for 16 rows
(about 28 per row). The suffix sums are moved window - 1 rows forward with an
in-register gather. Each program covers 2048 - (window - 1) outputs.

## Tests

n in {0, 1, 1000, 1000003} and window in {1, 2, 7, 1000, 1023, 1024}: dyadic
values, for which every sum is exact, must match the reference bit for bit;
values of magnitudes 1e-8 to 1e8 and mixed signs must stay within the error
bound (window - 1) * eps * sum(|x|) of the correctly rounded window sum
(`math.fsum`). The test also compiles the kernel with the declared `TYPES`.

## Performance (RTX A6000, n = 10^8)

window 1: 2.34 ms (683 GB/s, memory bound). Window 7 to 100: 3.8 to 4.0 ms.
Window 1024: 7.2 ms (half of each region is the halo). The first version, with
flat-tile scans, took 9.5 ms for any window above 1.
