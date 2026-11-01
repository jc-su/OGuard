# q17_cuda_q17_build_map_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q17_build_map_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q17_build_map_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, and `<algorithm>`
  includes, the macros `BLOCK_SIZE` and `MAX_BLOCKS_PER_SM`, and the host code,
  which the kernel does not use.

## Launch

`q17_launcher` launches the kernel once with
`<<<q17_blocks_for(n_parts), BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1), no
dynamic shared memory; `q17_blocks_for` caps the grid at 8 blocks per SM, and the
kernel strides over the rows.

## Lengths and ranges

- `part_keys`: the filtered `p_partkey` values, `n_parts = part_keys.numel()`
  rows. `run_query` sets `map_size = max(filtered part keys) + 1`, and part keys
  are positive, so every value lies in `[0, map_size - 1]`: `index-in` into
  `part_map`.
- `part_map`: `torch::empty({map_size})` of int32, set to -1 by
  `cudaMemsetAsync(0xff)` before the launch; the kernel only stores into it
  (`write`).
- `n_parts`, `map_size`: size symbols with least value 1; the launcher returns
  for `n_parts == 0` or `map_size <= 0`.
