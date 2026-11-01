# q07_triton__q7_agg_kernel

Source: `azure__gpt-5.5/triton/full/queries/q07/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q7_agg_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q7_agg_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q7_agg` (plan.py line 124) launches the kernel once with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `supp_code_ptr`, `cust_code_ptr` (int32 codes 0 or 1), `shipdate_days_ptr`
  (int32), `price_ptr`, `discount_ptr` (float64): the columns of the cuDF join
  result `joined`, `n_elements = extendedprice.numel()` rows.
- `out_ptr`: `torch.zeros(8, dtype=torch.float64)`, 8 elements, which the kernel
  adds to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1; the launcher returns when `n == 0`.
- `date_lo`, `date_hi`, `year_split_days`: Python `int` epoch days of
  1995-01-01, 1996-12-31, and 1996-01-01, `i32`; comparison bounds only.
