# tile_scan notes

- One block of 256 threads scans one tile, so the grid has
  `(n + 1023) / 1024` blocks, one per element of `totals`.
- The sums wrap around in 32-bit two's complement arithmetic. `kernel.cu`
  adds in `uint32_t` so that the wrap is defined behavior. The tests draw
  values from the whole int32 range, so the sums do wrap, and the references
  wrap the same way.
- compute-sanitizer memcheck, racecheck, synccheck, and initcheck report no
  errors for both versions on the test inputs.
