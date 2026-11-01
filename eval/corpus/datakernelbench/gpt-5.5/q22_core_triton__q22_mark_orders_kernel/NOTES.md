# q22_core_triton__q22_mark_orders_kernel

Source: `azure__gpt-5.5/triton/core/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q22_mark_orders_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q22_mark_orders_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q22` launches the kernel once, when `n_orders > 0`, with grid
`(triton.cdiv(n_orders, 1024),)`, `BLOCK_SIZE=1024`, `max_key=n_customers`, and
`num_warps=8`.

## Lengths and ranges

- `o_custkey_ptr`: the distinct `o_custkey` values of orders, `n_orders` rows (`i32`); a foreign key into `customer`, so `read` (the kernel bounds each key itself).
- `bitmap_ptr`: `torch.zeros(max_key + 1, dtype=torch.uint8)`, `max_key + 1` bytes, written only.
- `max_key`: `n_customers`, the customer row count (1,500,000, `i32`), a size symbol with least value 0 at this launch (`n_customers > 0` was checked before, so it is at least 1; the review uses `max_key >= 0`).
- `n_orders`: size symbol with least value 1.
