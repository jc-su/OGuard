# q22_core_triton__q22_avg_kernel

Source: `azure__gpt-5.5/triton/core/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q22_avg_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q22_avg_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q22` launches the kernel once, when `n_customers > 0`, with grid
`(triton.cdiv(n_customers, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=8`.

## Lengths and ranges

- `acctbal_ptr` (`float64`), `in_mask_ptr` (a `torch.bool` tensor, `*i1`): the customer columns, `n_elements = c_custkey.numel()` rows each (1,500,000 at scale factor 10, so `i32`).
- `stats_ptr`: `torch.zeros(2, dtype=torch.float64)`, 2 elements, added to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1; the launcher returns before the launch when it is 0.
