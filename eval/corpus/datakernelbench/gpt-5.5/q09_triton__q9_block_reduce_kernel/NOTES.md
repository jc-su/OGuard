# q09_triton__q9_block_reduce_kernel

Source: `azure__gpt-5.5/triton/full/queries/q09/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q9_block_reduce_kernel` of `_KERNEL_SOURCE`
(plan lines 18 to 58).

## Extraction edits

- Renamed `_q9_block_reduce_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan lines 154 to 170) launches the kernel when
`n_years <= 16`, with grid `(triton.cdiv(n, block_size),)`,
`N_YEARS=int(n_years)`, `BLOCK_SIZE=int(block_size)`, and `num_warps=8`.
`block_size` is the default 512 of `run_query` and `_query_core`. `N_YEARS`
is data dependent: `run_query` sets `min_year` and `max_year` to the least and
greatest `o_year` of the joined rows, and TPC-H order dates run from 1992 to
1998, so at scale factor 10 `n_years = 7`. The other branch launches
`_q9_row_atomic_kernel`.

## Lengths and ranges

- `extendedprice_ptr`, `discount_ptr`, `supplycost_ptr`, `quantity_ptr`
  (float64), `nationkey_ptr` (int64), `year_ptr` (int32): the columns of the
  joined frame `joined`, `n_elements` rows each. `n = l_extendedprice.numel()`,
  below 2^31 at scale factor 10, so `i32`.
- `n_elements`: size symbol with least value 1; `_query_core` returns before
  the launch when `n == 0`, and `run_query` returns when the join is empty.
- `sum_out_ptr`, `cnt_out_ptr`: `torch.zeros(25 * n_years)` float64, 175
  elements at `n_years = 7`; the kernel adds to them with `tl.atomic_add`
  (`readwrite`).
- `min_year`: a Python `int`, the least joined `o_year` (1992 at scale
  factor 10), `i32`; the kernel only subtracts it and compares the result.
