# q20_triton__build_ps_slots_kernel

Source: `azure__gpt-5.5/triton/full/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_build_ps_slots_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_build_ps_slots_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_build_partsupp_slots` (plan.py line 132) launches the kernel once, when
`n_ps > 0`, with grid `(n_ps,)`, one program per filtered `partsupp` row,
`max_slots=4` (the default of `run_query` and `_query_core`), and `num_warps=1`.

## Lengths and ranges

- `ps_partkey_ptr`, `ps_suppkey_ptr` (`int64`): the columns of
  `filtered_partsupp`, the inner join of `partsupp` with the parts whose name
  starts with `forest`, `n_ps` rows each; `n_ps` is at least 1 at the launch.
- `counts_ptr`: `torch.zeros((n_parts,), dtype=torch.int32)`, `n_parts` entries,
  where `n_parts = int(df_part["p_partkey"].max()) + 1` (2,000,001 at scale
  factor 10, at least 1). The kernel increments it with `tl.atomic_add` and uses
  the old value as a slot (`readwrite`).
- `supp_slots_ptr`, `idx_slots_ptr`: `torch.full((n_parts * max_slots,), -1)` of
  `int64`, `4 * n_parts` entries each; the kernel only stores to them (`write`).
- `ps_partkey_ptr` is `index-in` into `counts_ptr`: after the inner join each value
  is a `p_partkey` of `part`, and `n_parts` is the maximum of `p_partkey` plus 1,
  with positive keys, so every value lies in `[0, n_parts - 1]`.
