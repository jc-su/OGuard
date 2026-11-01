# q18_core_triton__dense_order_sum_kernel

Source: `azure__gpt-5.5/triton/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_dense_order_sum_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_dense_order_sum_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_dense_order_sums` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and no `num_warps` (4).

## Lengths and ranges

- `orderkey_ptr` (`int64`), `quantity_ptr` (`float64`): `l_orderkey` and `l_quantity`, `n_elements = l_orderkey.numel()` rows each (59,986,052 at scale factor 10, so `i32`).
- `sums_ptr`: `torch.zeros(table_size, dtype=torch.float64)`, `table_size = max(l_orderkey) + 1` elements, added to with `tl.atomic_add` (`readwrite`); `table_size` is a size symbol that the kernel does not receive, least value 1. It is sized from the maximum of the same keys, and TPC-H keys are positive, so `orderkey_ptr` is `index-in` `sums_ptr`.
- `n_elements`: size symbol with least value 1; `_query_core` returns before the launch when `n == 0`.
