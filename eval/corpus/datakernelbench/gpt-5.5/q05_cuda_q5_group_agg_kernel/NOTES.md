# q05_cuda_q5_group_agg_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q05/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q5_group_agg_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q5_group_agg_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>` and the macro `MAX_GROUPS`; left out the Torch,
  ATen, c10, and `cuda_runtime.h` includes, the macro `BLOCK_SIZE` (used only by
  the host code), and the host code.

## Launch

`q5_aggregate_launcher` launches the kernel when `n > 0 && num_nations > 0` with
`<<<(n + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `nation_id_ptr`, `price_ptr`, `discount_ptr`: `n_elements` rows each, the
  joined table; the launcher sets `n = price.numel()` and checks the other two.
- `out_ptr`: `torch::zeros({num_nations64})`, `num_nations` elements, which the
  kernel adds to with `atomicAdd` (`readwrite`).
- `nation_id_ptr`: `nation_id` is `range(num_nations)` on the region's nations,
  carried through the joins, so every value is in [0, num_nations - 1]:
  `index-in` into `out_ptr`.
- `num_nations`: the region's nation count; `TORCH_CHECK` bounds it by
  `MAX_GROUPS` (32) and the launch needs it positive: least value 1,
  `size_maxima` 32 in `launch.json`. At scale factor 10 it is 5.
- `n_elements`: size symbol with least value 1.
