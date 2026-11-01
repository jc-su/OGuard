# block_sort

- One block of 256 threads sorts one tile of 1024 keys with a bitonic sort in
  shared memory (55 compare-exchange steps, 2 pairs per thread per step).
  `run` launches `(n + 1023) / 1024` blocks (at least one). A block whose tile
  index is past the last tile returns, so a grid of `(n + 255) / 256` blocks
  is also correct.
- A short last tile is padded with `INT32_MAX` in shared memory. The padding
  sorts after every real key, so the first `len` slots hold the tile's keys in
  order even when a real key equals `INT32_MAX`. Only the real rows are
  written back.
