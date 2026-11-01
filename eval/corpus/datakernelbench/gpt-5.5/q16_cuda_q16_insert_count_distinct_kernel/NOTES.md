# q16_cuda_q16_insert_count_distinct_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q16/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q16_insert_count_distinct_kernel` of
`_CUDA_SOURCE`.

## Extraction edits

- Renamed `q16_insert_count_distinct_kernel` to `kernel` and declared it
  `extern "C"`.
- Kept `#include <stdint.h>`, the macro `EMPTY_KEY`, and the device function
  `mix64`; left out the Torch, ATen, c10, `cuda.h`, and `cuda_runtime.h`
  includes, the macro `BLOCK_SIZE`, and the host code, which the kernel does not use.

## Launch

`q16_count_distinct_launcher` launches the kernel once with
`<<<(n + 255) / 256, 256, 0, stream>>>`: block (256, 1, 1), no dynamic shared
memory. Before it, `fill_i64_kernel` zeroes `counts` and fills `hash_table`
with -1 (`EMPTY_KEY`).

## Lengths and ranges

- `group_ids`, `supp_ids`: `n` rows each; the launcher sets `n = group_ids.numel()`
  and checks `group_ids.numel() == supp_ids.numel()`. `group_ids` is the `_gid`
  column of `joined_with_gid`, an inner merge with `group_df`, whose `_gid` is
  `range(len(group_df))`, so every value lies in `[0, num_groups - 1]`:
  `index-in` into `counts`.
- `n`: size symbol with least value 1; the launcher returns for `n == 0`.
- `hash_table`: `torch::empty({hash_size})` of int64 passed as
  `unsigned long long*`, `hash_size` a power of two at least 16; the kernel
  CASes and reads it (`readwrite`). `hash_mask = hash_size - 1` is the size
  symbol, so the length is `hash_mask + 1` and the least `hash_mask` is 15.
- `counts`: `torch::empty({num_groups})`, zeroed by `fill_i64_kernel`, which the
  kernel adds to with `atomicAdd` (`readwrite`).
- `num_groups`: `len(group_df)`, size symbol with least value 1; the launcher
  returns for `num_groups == 0`, and `run_query` returns before it for 0.
