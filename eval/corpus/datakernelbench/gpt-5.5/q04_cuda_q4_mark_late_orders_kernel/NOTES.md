# q04_cuda_q4_mark_late_orders_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q04/round_03.py` (`output_round` 3),
copied as `plan.py`. The kernel is `q4_mark_late_orders_kernel` of
`_CUDA_SOURCE`.

## Extraction edits

- Renamed `q4_mark_late_orders_kernel` to `kernel` and declared it
  `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, and c10 includes, the
  macros `BLOCK_SIZE` and `MAX_PRIORITIES` (not used by this kernel), the other
  kernel, and the host code.

## Launch

`q4_launcher` launches the kernel when `n_lineitem > 0` with
`<<<(n_lineitem + 1023) / 1024, BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1),
no dynamic shared memory.

## Lengths and ranges

- `l_orderkey`, `l_commitdate_days`, `l_receiptdate_days`: `n_lineitem` rows
  each; `n_lineitem = l_orderkey.numel()` and the launcher checks the two date
  columns against it.
- `late_bitmap`: `torch::zeros({bitmap_size})` bytes with
  `bitmap_size = max_lineitem_orderkey + 1`, which the kernel only stores 1 into
  (`write`).
- `l_orderkey`: `max_lineitem_orderkey` is `l_orderkey.max()` of the same
  column, so every key is at most `bitmap_size - 1`, and TPC-H keys are
  positive: `index-in` into `late_bitmap`.
- `n_lineitem`: size symbol with least value 1 (the launch needs it positive).
- `bitmap_size`: size symbol with least value 1 (`TORCH_CHECK` of
  `max_lineitem_orderkey >= 0`).
