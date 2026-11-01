# q13_cuda_reduce_max_array_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q13/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `reduce_max_array_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `reduce_max_array_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch
  and CUDA runtime includes, `<algorithm>`, `<vector>`, the macro
  `CUDA_CHECK_LAST`, the device functions, and the host code, which the
  kernel does not use.

## Launch

`q13_launcher` launches the kernel in the loop `while (cur_n > 1)`, each time
with `<<<next_blocks, BLOCK_SIZE>>>`, `next_blocks = launch_blocks_for(cur_n) =
min((cur_n + 1023) / 1024, 131072)`: block (1024, 1, 1), no dynamic shared
memory. Every launch of the loop has this configuration, so it is one corpus
kernel. At scale factor 10 (1,500,000 customers) the loop launches it twice,
for 1465 and for 2 values.

## Lengths and ranges

- `values`: `cur`, the `cur_n` block maxima of the previous launch
  (`torch::empty({first_blocks})` or `torch::empty({next_blocks})`), `n_values`
  elements.
- `n_values`: `cur_n`, size symbol with least value 2; the loop launches only
  while `cur_n > 1`.
- `block_max`: `torch::empty({next_blocks})`, one element per block of the
  plan's grid, `min((n_values + 1023) / 1024, 131072)`. The signature states
  the quotient `(n_values + 1023) / 1024`, which gives the buffer its own size
  symbol; its least value is 1. The kernel only writes it.
