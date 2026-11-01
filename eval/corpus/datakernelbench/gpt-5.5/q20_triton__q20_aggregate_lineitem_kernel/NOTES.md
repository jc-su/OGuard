# q20_triton__q20_aggregate_lineitem_kernel

Source: `azure__gpt-5.5/triton/full/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q20_aggregate_lineitem_kernel` of
`_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q20_aggregate_lineitem_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan.py line 180) launches the kernel once, after
`_build_ps_slots_kernel`, with grid `(triton.cdiv(n_lineitem, 256),)`,
`max_slots=4`, `BLOCK_SIZE=256` (the defaults of `run_query`), and `num_warps=4`.

## Lengths and ranges

- `l_partkey_ptr`, `l_suppkey_ptr` (`int64`), `l_quantity_ptr` (`float64`),
  `l_shipdate_days_ptr` (`int32`): columns of `lineitem`, `n_lineitem` rows each;
  `n_lineitem = l_partkey.numel()`, and `_query_core` returns before the launch
  when it is 0, so its least value is 1.
- `counts_ptr` (`int32`, `n_parts` entries), `supp_slots_ptr`, `idx_slots_ptr`
  (`int64`, `4 * n_parts` entries): the buffers that `_build_ps_slots_kernel`
  filled; `n_parts = max(p_partkey) + 1`, 2,000,001 at scale factor 10, so `i32`.
  `idx_slots_ptr` holds row numbers of `filtered_partsupp` or the sentinel -1, so
  its role is `read`.
- `sum_qty_ptr`: `torch.zeros((n_ps,), dtype=torch.float64)`, one entry per
  filtered `partsupp` row, which the kernel adds to with `tl.atomic_add`
  (`readwrite`); `n_ps` is at least 1 at the launch.
- `l_partkey_ptr` is `read`: its values are foreign keys into `part`; the kernel
  checks `0 <= p < n_parts` before it reads the slots.
- `shipdate_lo_days`, `shipdate_hi_days`: Python `int` epoch days, `i32`;
  comparison thresholds only.
