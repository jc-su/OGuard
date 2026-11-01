# q11_triton__q11_scan_kernel

Source: `azure__gpt-5.5/triton/full/queries/q11/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q11_scan_kernel` of `_KERNEL_SOURCE`
(plan lines 17 to 47).

## Extraction edits

- Renamed `_q11_scan_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_scan` (plan lines 127 to 167) launches the kernel once with grid
`(triton.cdiv(n, block_size),)`, `supplier_mask_n=supplier_mask.numel()`,
`BLOCK_SIZE=block_size` (the default 1024 of `_query_core`), and no
`num_warps` (4). `supplier_mask_n` is a `tl.constexpr` whose value is
`max(s_suppkey) + 1`, 100,001 at scale factor 10 (100,000 suppliers).

## Lengths and ranges

- `ps_partkey_ptr`, `ps_suppkey_ptr`, `ps_availqty_ptr` (int64),
  `ps_supplycost_ptr` (float64): the columns of `partsupp`, `n_elements` rows
  each; `n = ps_suppkey.numel()`, `i32`.
- `n_elements`: size symbol with least value 1; `run_query` returns for an
  empty `partsupp` and `_launch_scan` returns when `n == 0`.
- `supplier_mask_ptr`: `torch.zeros(max_suppkey + 1)` int8, 100,001 elements at
  scale factor 10, the value of `supplier_mask_n`.
- `group_values_ptr`: `torch.zeros(max_partkey + 1)` float64 with
  `max_partkey = max(ps_partkey)`; its length is the symbol `n_groups` (least
  value 1). The kernel adds to it with `tl.atomic_add` (`readwrite`).
- `ps_partkey_ptr` is `index-in` into `group_values_ptr`: the plan sizes
  `group_values` from the maximum of the same column, and TPC-H part keys are
  positive, so every value lies in `[0, n_groups - 1]`.
- `total_ptr`: `torch.zeros(1)` float64, added to with `tl.atomic_add`
  (`readwrite`).
