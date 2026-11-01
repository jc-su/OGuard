# q20_core_triton__insert_qual_supp_kernel

Source: `azure__gpt-5.5/triton/core/queries/q20/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `_insert_qual_supp_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_insert_qual_supp_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q20_triton` launches the kernel once with grid `(n_ps,)`, one program per
filtered partsupp row, `MAX_PROBE=64`, `table_mask=supp_table_size - 1`, and
`num_warps=1`.

## Lengths and ranges

- `ps_suppkey_ptr`, `ps_availqty_ptr` (`int64`), `sum_qty_ptr` (`float64`, the sums of `_accum_lineitem_kernel`): `n_ps` elements each; read only. `n_ps` is a size symbol that the kernel does not receive, least value 1.
- `supp_set_ptr`: `torch.full((supp_table_size,), -1)` of `int64`, `supp_table_size = table_mask + 1` entries, probed with `tl.atomic_cas` (`readwrite`).
- `table_mask`: `supp_table_size - 1` with `supp_table_size` the smallest power of two of at least `max(16, 8 * n_ps)` (`i32`): size symbol with least value 15.
