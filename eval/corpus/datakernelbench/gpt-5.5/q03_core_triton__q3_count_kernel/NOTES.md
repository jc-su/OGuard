# q03_core_triton__q3_count_kernel

Source: `azure__gpt-5.5/triton/core/queries/q03/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q3_count_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q3_count_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_compact_q3_rows` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `l_shipdate_days_ptr`, `o_orderdate_days_ptr`: `int32` epoch days of the joined rows, `n_elements = l_orderkey.numel()` rows each (below 2^31, so `i32`).
- `keep_ptr`: `torch.empty((n,), dtype=torch.uint8)`, `n_elements` bytes, written only.
- `counts_ptr`: `torch.empty((num_blocks,), dtype=torch.int64)`, one element per program of the plan's grid, `(n_elements + 1023) / 1024`; the quotient gives the buffer its own size symbol with least value 1. Written only.
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
- `shipdate_lo_days`, `orderdate_hi_days`: Python `int` epoch days, `i32`; comparison thresholds only.
