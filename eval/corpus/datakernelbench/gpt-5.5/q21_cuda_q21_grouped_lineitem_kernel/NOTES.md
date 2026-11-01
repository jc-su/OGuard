# q21_cuda_q21_grouped_lineitem_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q21/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q21_grouped_lineitem_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q21_grouped_lineitem_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`; left out the Torch, ATen, and c10 includes, the
  macro `BLOCK_SIZE` (not used by the kernel), and the host code.

## Launch

`q21_grouped_launcher` launches the kernel once, when `n_line`, `n_orders`,
`n_supp`, and `num_suppliers` are all positive, with
`<<<(n_line + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no
dynamic shared memory, after the two map kernels on the same stream.

## Lengths and ranges

- `l_orderkey`, `l_suppkey`, `l_receipt`, `l_commit`: `n_line` rows each, the
  columns of `lineitem` (the dates as int64 seconds). The launcher takes
  `n_line = l_orderkey.numel()`; the four come from the same table.
- `n_line`: size symbol with least value 1.
- `order_flags`: the `max_orderkey + 1` bytes of `q21_set_order_flags_kernel`;
  `max_orderkey` size symbol with least value 0.
- `supp_map`: the `max_suppkey + 1` int32 entries of
  `q21_set_supplier_map_kernel` (-1 or a supplier position); `max_suppkey` size
  symbol with least value 0. The kernel checks each value against
  `num_suppliers` before using it, so its role is `read`.
- `num_suppliers`: `len(supp_n)`, size symbol with least value 1 (`run_query`
  returns for 0 and the launch requires it positive).
- `counts`: `torch::zeros({num_suppliers})` of int64, which the kernel increments
  with `atomicAdd` (`readwrite`).
- The kernel relies on `lineitem` being grouped by `l_orderkey` for its
  result, not for its accesses.
