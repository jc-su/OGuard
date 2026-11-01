# q13_core_triton__scatter_order_counts_i32_kernel

Source: `azure__gpt-5.5/triton/core/queries/q13/round_06.py` (`output_round` 6),
copied as `plan.py`. The kernel is `_scatter_order_counts_i32_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_scatter_order_counts_i32_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, when `n_orders > 0`, with grid
`(ceil(n_orders / 256),)`, `BLOCK_SIZE=256`, and `num_warps=4`.

## Lengths and ranges

- `o_custkey_ptr`: the customer keys of the orders without special requests, `n_orders` rows (below 2^31, so `i32`); a foreign key into `customer`, so `read`.
- `counts_ptr`: the `counts_n` `int32` entries that `_zero_i32_counts_and_hist_kernel` cleared, added to with `tl.atomic_add` (`readwrite`). `counts_n = max(c_custkey) + 1` is a size symbol that the kernel does not receive, least value 1.
- `n_orders`: size symbol with least value 1; the launch runs only when `n_orders > 0`.
