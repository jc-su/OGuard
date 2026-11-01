# valid_rows (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

Each program owns 512 bitmap bytes (4096 rows). It clears the padding bits
past row n, counts the valid rows of each byte with a popcount, scans the byte
counts, and reserves the tile's output range with one atomic add on `count`.
Each valid row id goes to tile base + byte prefix + popcount of the lower bits
of its byte, on a [512, 8] tile whose lanes map to consecutive rows, so the
stores of a warp cover a contiguous output range. Ids are ascending inside a
tile; tiles land in atomic order (the task allows any order).

## Tests

n in {0, 1, 1000, 1000003}, densities 0, 0.03, 0.5 and 1, each with the
padding bits of the last byte clear and set. The test also compiles the kernel
with the declared `TYPES` (int64 scalars).

## Performance (RTX A6000, n = 10^8)

Density 1.0: 618 us, 668 GB/s. Density 0.5: 331 us, 641 GB/s. Density 0.03:
109 us; this case is bound by the per-row instructions of the [512, 8] tile.
A variant that loops only over the set bits of each byte was slower above
density 0.001 because its stores are not coalesced.
