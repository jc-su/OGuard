# block_sort (Triton)

- One program per 1024-row tile, sorted with `tl.sort` (a bitonic sort over
  the 1024-element block). A short last tile is padded with INT32_MAX
  (2147483647) on load and stored under the row mask; since the padding is at
  least every real key, the first `len` sorted positions hold exactly the real
  keys, including keys that equal INT32_MAX.
- 4 warps (8 keys per thread); 2, 4, and 8 warps all run in about 31 us for
  n = 1000003 on the RTX A6000, 16 warps takes 39 us.
