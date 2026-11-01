# q02_triton__partsupp_filter_kernel

Source: `azure__gpt-5.5/triton/full/queries/q02/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_partsupp_filter_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_partsupp_filter_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_partsupp_filter` (plan.py line 118) launches the kernel once with grid
`(triton.cdiv(n_rows, 256),)`, `BLOCK_SIZE=256`, and `num_warps=4`.

## Lengths and ranges

- `ps_partkey_ptr`, `ps_suppkey_ptr`: int64 partsupp columns, `n_rows =
  ps_partkey.numel()` rows (8,000,000 at scale factor 10, so `i32`).
- `part_flags_ptr`, `supp_flags_ptr`: the int8 flag tensors that
  `_mark_keys_kernel` filled, of `n_part_flags = part_flags.numel()` and
  `n_supp_flags = supp_flags.numel()` elements. The partsupp keys are not bounded
  by these lengths by the plan (they are foreign keys into part and supplier), so
  both key columns are `read`; the kernel bounds each key itself.
- `out_mask_ptr`: `torch.empty((n_rows,), dtype=torch.bool)`, which the kernel
  only stores into (`write`); Triton's type of a bool tensor is `*i1`.
- `n_rows`: size symbol with least value 1; the launcher launches only when
  `n_rows > 0`. `n_part_flags`, `n_supp_flags`: `max(key) + 1` of nonempty
  positive key sets, least value 1.
