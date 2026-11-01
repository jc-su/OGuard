# q01_triton__q1_group_kernel

Source: `azure__gpt-5.5/triton/full/queries/q01/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q1_group_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q1_group_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q1_group_kernel` (plan.py line 126) launches the kernel once with grid
`(triton.cdiv(n_elements, BLOCK_SIZE),)`, `BLOCK_SIZE=1024` (the default
`block_size`), `num_warps=8`, and the constexprs `N_RF` and `N_LS`, which are
the category counts of `l_returnflag` and `l_linestatus` that cuDF's
`astype("category")` finds. At scale factor 10 these are 3 (`A`, `N`, `R`) and
2 (`F`, `O`), as the benchmark's summary of lineitem at sf10 states, so
`N_RF=3` and `N_LS=2`.

## Lengths and ranges

- `rf_ptr`, `ls_ptr`: int32 category codes of `l_returnflag` and `l_linestatus`,
  `n_elements` rows each; `qty_ptr`, `price_ptr`, `disc_ptr`, `tax_ptr`: float64
  lineitem columns; `shipdate_ptr`: int32 epoch days. All are lineitem columns of
  `n_elements = l_quantity.numel()` rows, below 2^31 at scale factor 10, so `i32`.
- `n_elements`: size symbol with least value 1; the launcher returns before the
  launch when `n_elements == 0` or a category count is not positive.
- `out_ptr`: `torch.zeros(n_groups * 7)` with `n_groups = max(N_RF * N_LS, 1) = 6`,
  so 42 float64 elements, which the kernel adds to with `tl.atomic_add`
  (`readwrite`).
- `date_hi`: Python `int` epoch day of 1998-09-02, `i32`; a comparison bound only.
