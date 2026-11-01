# q09_core_triton__q9_amount_kernel

Source: `azure__gpt-5.5/triton/core/queries/q09/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q9_amount_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q9_amount_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q9_amount` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `l_extendedprice_ptr`, `l_discount_ptr`, `ps_supplycost_ptr`, `l_quantity_ptr`: the joined columns, `n_elements = l_extendedprice.numel()` rows each (below 2^31, so `i32`).
- `out_ptr`: `torch.empty_like(l_extendedprice)`, `n_elements` elements, written only.
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
