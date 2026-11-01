# block_sort (Triton)

- `NUM_WARPS` is 1: one warp holds a 1024-key tile as 32 keys per thread, so
  more of the bitonic network runs within a thread and less of it exchanges
  keys through shared memory. Time on GPU 1 (RTX A6000) for 2^25 keys:
  1 warp 569 us, 2 warps 699 us, 4 warps 786 us, 8 warps 911 us, 16 warps
  1191 us.
