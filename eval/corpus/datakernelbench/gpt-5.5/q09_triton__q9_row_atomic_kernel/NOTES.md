# q09_triton__q9_row_atomic_kernel

Source: `azure__gpt-5.5/triton/full/queries/q09/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q9_row_atomic_kernel` of `_KERNEL_SOURCE`
(plan lines 61 to 93).

## Extraction edits

- Renamed `_q9_row_atomic_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan lines 171 to 188) launches the kernel when
`n_years > 16`, with grid `(triton.cdiv(n, 256),)`, `BLOCK_SIZE=256`, and
`num_warps=4`. At scale factor 10 the order years span 1992 to 1998
(`n_years = 7`), so the plan takes the other branch
(`_q9_block_reduce_kernel`); this launch is the plan's branch for a wider
year span.

## Lengths and ranges

- `extendedprice_ptr`, `discount_ptr`, `supplycost_ptr`, `quantity_ptr`
  (float64), `nationkey_ptr` (int64), `year_ptr` (int32): the columns of the
  joined frame, `n_elements` rows each; `n = l_extendedprice.numel()`, `i32`.
- `n_elements`: size symbol with least value 1; `_query_core` returns before
  the launch when `n == 0`.
- `n_years`: `max_year - min_year + 1`, a size symbol with least value 17 on
  this branch; a Python `int`, `i32`.
- `sum_out_ptr`, `cnt_out_ptr`: `torch.zeros(25 * n_years)` float64, which the
  kernel adds to with `tl.atomic_add` (`readwrite`).
- `min_year`: a Python `int`, the least joined `o_year`, `i32`; the kernel only
  subtracts it and compares the result.
