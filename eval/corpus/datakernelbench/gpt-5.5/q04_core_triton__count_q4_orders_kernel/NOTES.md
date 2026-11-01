# q04_core_triton__count_q4_orders_kernel

Source: `azure__gpt-5.5/triton/core/queries/q04/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_count_q4_orders_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_count_q4_orders_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q4_triton` launches the kernel once, when `n_orders > 0` and
`num_priorities > 0`, with grid `(triton.cdiv(n_orders, 1024),)`, `BLOCK_SIZE=1024`,
`NUM_PRIORITIES=num_priorities` (a `tl.constexpr`, the 5 order priorities of TPC-H),
and no `num_warps` (4).

## Lengths and ranges

- `o_orderkey_ptr` (`int64`), `o_orderdate_days_ptr`, `o_orderpriority_codes_ptr` (`int32`): the orders columns, `n_elements = o_orderkey.numel()` rows each (15,000,000 at scale factor 10, so `i32`).
- `o_orderkey_ptr`: a foreign key probed against the lineitem flag table, so `read`; the kernel bounds each key itself.
- `flags_ptr`: the `flag_n` bytes that `_mark_late_orders_kernel` set; read only. `flag_n = max(l_orderkey) + 1`, size symbol with least value 1 (`i32`).
- `counts_ptr`: `torch.zeros(num_priorities, dtype=torch.int64)`, 5 elements, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1. `date_lo_days`, `date_hi_days`: `i32` comparison thresholds only.
