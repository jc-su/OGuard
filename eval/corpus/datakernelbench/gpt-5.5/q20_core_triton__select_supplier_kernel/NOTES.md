# q20_core_triton__select_supplier_kernel

Source: `azure__gpt-5.5/triton/core/queries/q20/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `_select_supplier_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_select_supplier_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q20_triton` launches the kernel once with grid `(triton.cdiv(n_s, 256),)`,
`BLOCK_SIZE=256`, `MAX_PROBE=64`, `table_mask=supp_table_size - 1`, and `num_warps=4`.

## Lengths and ranges

- `s_suppkey_ptr`, `s_row_idx_ptr`: the keys and row numbers of the suppliers of the nation, `n_supplier` rows each (`i32`); read only (the row numbers are stored, not used as indices).
- `supp_set_ptr`: the `table_mask + 1` entries that `_insert_qual_supp_kernel` filled; read only.
- `out_idx_ptr`: `torch.empty((n_s,), dtype=torch.int64)`, `n_supplier` elements, written only; the plan slices the first `count` afterwards.
- `count_ptr`: `torch.zeros((1,), dtype=torch.int64)`, one element, the atomic cursor (`readwrite`).
- `n_supplier`: size symbol with least value 1; `table_mask`: a power of two minus one of at least 15.
