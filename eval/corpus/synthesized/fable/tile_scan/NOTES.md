# tile_scan

- One block of 256 threads handles one tile of 1024 rows: each thread loads 4
  consecutive rows, the per-thread sums are scanned with warp shuffles, and
  warp 0 scans the 8 warp totals through shared memory. `run` launches
  `(n + 1023) / 1024` blocks (at least one). A block whose tile index is past
  the last tile returns, so a grid of `(n + 255) / 256` blocks is also correct.
- The tile size is tied to 256 threads per block times 4 rows per thread.
