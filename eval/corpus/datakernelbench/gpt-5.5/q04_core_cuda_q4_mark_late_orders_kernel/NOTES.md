# q04_core_cuda_q4_mark_late_orders_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q04/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q4_mark_late_orders_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q4_mark_late_orders_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out `<algorithm>`, `<vector>`, the macro `Q4_BLOCK_SIZE`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q4_launcher` launches the kernel once, when `n_orders > 0` and `n_lineitem > 0`,
with `<<<(n_lineitem + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE, Q4_BLOCK_SIZE, 0, stream>>>`:
block (256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `l_orderkey` (`int64`), `l_commitdate_days`, `l_receiptdate_days` (`int32`): the lineitem columns, `n_lineitem = l_orderkey.numel()` rows each.
- `has_late_lineitem`: `torch::zeros({max_key + 1})` of `uint8`, `max_key + 1` bytes, which the kernel only writes.
- `max_key`: `max(max(o_orderkey), max(l_orderkey))`, checked `>= 0`; a size symbol with least value 0. The plan sizes `has_late_lineitem` from the maximum of the same `l_orderkey` values (and of `o_orderkey`), and TPC-H keys are positive, so `l_orderkey` is `index-in` `has_late_lineitem`.
- `n_lineitem`: size symbol with least value 1; the launcher returns before the launch when it is 0.
