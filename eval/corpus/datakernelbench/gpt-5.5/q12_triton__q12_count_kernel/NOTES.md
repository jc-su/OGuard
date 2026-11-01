# q12_triton__q12_count_kernel

Source: `azure__gpt-5.5/triton/full/queries/q12/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q12_count_kernel` of `_KERNEL_SOURCE`
(plan lines 18 to 48).

## Extraction edits

- Renamed `_q12_count_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q12_counts` (plan lines 120 to 142), which `run_query` calls,
launches the kernel once with grid `(triton.cdiv(n, BLOCK_SIZE),)`,
`BLOCK_SIZE=_BLOCK_SIZE` (4096), and `num_warps=8`.

## Lengths and ranges

- `code_ptr`, `high_ptr`: `shipmode_t` and `priority_t`, int32 columns of the
  lineitem and orders join `joined`, `n_elements` rows each;
  `n = l_shipmode_code.numel()`, `i32`.
- `n_elements`: size symbol with least value 1; the launcher returns before the
  launch when `n == 0`.
- `out_ptr`: `torch.zeros(4)` int64, added to at indices 0 to 3 with
  `tl.atomic_add` (`readwrite`).
