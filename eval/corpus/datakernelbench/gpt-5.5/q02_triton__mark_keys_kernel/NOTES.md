# q02_triton__mark_keys_kernel

Source: `azure__gpt-5.5/triton/full/queries/q02/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_mark_keys_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_mark_keys_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_mark_keys` (plan.py line 97) launches the kernel with grid
`(triton.cdiv(n_keys, 256),)`, `BLOCK_SIZE=256` (`block_size` of `run_query`),
and `num_warps=4`. `_filter_partsupp_triton` calls it twice with the same
configuration: once for the qualifying `p_partkey` values and once for the
European `s_suppkey` values. Both launches are this corpus kernel.

## Lengths and ranges

- `keys_ptr`: an int64 tensor of the keys, `n_keys = keys.numel()` rows.
- `flags_ptr`: `torch.zeros((n_flags,), dtype=torch.int8)`, `n_flags` elements,
  which the kernel only stores 1 into (`write`).
- `n_flags`: `int(keys.max()) + 1`, computed by `_filter_partsupp_triton` from the
  same keys (`part_keys.max()` or `supp_keys.max()`). TPC-H keys are positive, so
  every key lies in `[0, n_flags - 1]`: `keys_ptr` is `index-in` `flags_ptr`.
- `n_keys`, `n_flags`: size symbols with least value 1; the launcher launches only
  when `n_keys > 0 and n_flags > 0`. Both are below 2^31 at scale factor 10
  (`n_flags` at most 2,000,001), so `i32`.
