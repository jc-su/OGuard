# q22_core_triton__q22_agg_kernel

Source: `azure__gpt-5.5/triton/core/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q22_agg_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q22_agg_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q22` launches the kernel once, when `n_customers > 0`, with grid
`(triton.cdiv(n_customers, 1024),)`, `BLOCK_SIZE=1024`, `max_key=n_customers`, and
`num_warps=8`.

## Lengths and ranges

- `c_custkey_ptr` (`int64`), `acctbal_ptr` (`float64`), `cntrycode_ptr` (`int32`), `in_mask_ptr` (`torch.bool`, `*i1`): the customer columns, `n_elements` rows each (`i32`). `max_key` is the row count, not the maximum of the keys, so `c_custkey_ptr` is `read`; the kernel bounds each key itself.
- `bitmap_ptr`: the `max_key + 1` bytes that `_q22_mark_orders_kernel` set; read only. `max_key = n_customers` (`i32`), size symbol with least value 0.
- `stats_ptr`: the 2 sums of `_q22_avg_kernel`; read only.
- `counts_ptr` (`int64`), `sums_ptr` (`float64`): `torch.zeros(100)` each, 100 elements, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1.
