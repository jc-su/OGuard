# ascii_upper notes

- Both kernels convert the bytes `chars[offsets[0]:offsets[n]]` and leave
  `out[0:offsets[0]]` unwritten. The tests also run with `offsets[0] = 5`, as
  for a sliced Arrow column, and check that the bytes before `offsets[0]`
  keep their old values.
- `kernel.cu` loads and stores aligned 8-byte words and converts the 8 bytes
  of a word with bitwise operations. It requires `chars` and `out` to be
  8-byte aligned, which Arrow requires of its buffers. The partial words at
  the two ends of the range are converted byte by byte, so the kernel never
  touches a byte outside the range. The grid needs one thread per 8-byte word
  that overlaps the range; the kernel loops grid-stride, so a smaller grid is
  also correct.
- `numba.py` converts one byte per step, since a `uint8[:]` Numba array cannot
  be read as 8-byte words inside the kernel. `run` sizes the grid from
  `out.shape[0]` (which equals `offsets[n]`) at 16 bytes per thread, so the
  host does not read `offsets[n]` back. One byte per thread launched so many
  blocks that block scheduling limited the throughput to about half.
- Measured on the RTX A6000 with 16M rows (134 MB of bytes): `kernel.cu`
  0.40 ms (675 GB/s of reads and writes), `numba.py` 0.50 ms (537 GB/s).
