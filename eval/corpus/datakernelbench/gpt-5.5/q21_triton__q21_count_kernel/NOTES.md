# q21_triton__q21_count_kernel

Source: `azure__gpt-5.5/triton/full/queries/q21/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q21_count_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q21_count_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q21_count` (plan.py line 122) launches the kernel once with grid
`(triton.cdiv(n_cand, 256),)`, `BLOCK_SIZE=256` (the default `block_size`), and
`num_warps=8`.

## Lengths and ranges

- `cand_orderkey_ptr`, `cand_supp_id_ptr`, `cand_late_count_ptr` (`int64`): the
  columns of `candidates`, the late (order, supplier) pairs of the supplier
  nation, `n_cand` rows each.
- `stat_orderkey_ptr`, `stat_distinct_ptr`, `stat_late_distinct_ptr` (`int64`):
  the per-order statistics, sorted by `l_orderkey`, `n_stats` rows each.
- `counts_ptr`: `torch.zeros(num_suppliers, dtype=torch.int64)`, which the kernel
  adds to with `tl.atomic_add` (`readwrite`).
- `n_cand`, `n_stats`, `num_suppliers`: the launcher returns before the launch
  when any of them is 0, so each is at least 1.
- `cand_supp_id_ptr` is `index-in` into `counts_ptr`: `supp_id` is
  `cudf.Series(range(num_suppliers))` on `supp_n`, and the inner merge keeps only
  those values, so every value lies in `[0, num_suppliers - 1]`.
- The binary search over `stat_orderkey_ptr` needs the rows sorted; the review
  treats sortedness as a fact the gate does not have.
