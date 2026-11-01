# ascii_upper (Triton)

- The kernel is elementwise over bytes, not rows. It reads the byte count as
  `offsets[n]` on the device; `run` sizes the grid from `out.numel()`, which is
  the same number.
- BLOCK=1024 bytes per program with 4 warps (8 bytes per thread). Benchmark at
  12 MB of chars on the RTX A6000: 40 us (about 600 GB/s for the read plus the
  write). BLOCK=4096 with 4 warps took 91 us and BLOCK=8192 took 80 to 95 us.
