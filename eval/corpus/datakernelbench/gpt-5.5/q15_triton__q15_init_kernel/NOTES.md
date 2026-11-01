# q15_triton__q15_init_kernel

Source: `azure__gpt-5.5/triton/full/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q15_init_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q15_init_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once with grid `(triton.cdiv(n_sup, 1024),)`,
`n_sup` as a `tl.constexpr`, `BLOCK_SIZE=1024`, and `num_warps=4`.
`n_sup = s_suppkey.numel()`, the rows of `supplier`: 100,000 at scale factor 10
(the benchmark's `meta_data/sf10/summarize/supplier.csv` counts 100000), so
`CONSTANTS["n_sup"] = 100000`.

## Lengths and ranges

- `totals_ptr` (`float64`), `present_ptr` (`uint8`): `torch.empty(n_sup)`,
  100000 elements each, which the kernel only writes.
- `count_ptr`: `torch.empty(1)` of `int32`, one element, which program 0 writes.
- The lengths are constants because `n_sup` is a constexpr of this
  configuration; the plan returns before the launch when `n_sup == 0`.
