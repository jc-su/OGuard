# q17_core_triton__q17_filter_sum_kernel

`triton.py` is the kernel `_q17_filter_sum_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q17/round_05.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q17_filter_sum_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The `@triton.autotune` decorator is replaced by the values of this configuration, `BLOCK_SIZE = 4096`, 4 warps. The autotuner lists 12 configurations: `BLOCK_SIZE` 4096, 8192, 16384, `num_warps` 4 and 8, `num_stages` 2 and 4. The module states `CONSTANTS` and `NUM_WARPS` and no `num_stages`, and the kernel has no loop that Triton pipelines (the front end's PTX is the same for `num_stages` 2, 4, and Triton's default), so the two configurations that differ only in `num_stages` are one corpus kernel. The six kernels are the configurations in the autotuner's order, the first without a suffix and the others `_2` to `_6`; this is number 1.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_query_core` launches `_q17_filter_sum_kernel[(cdiv(n, BLOCK_SIZE),)]` once, through the autotuner, when `keep_mask` has a true element.

## Lengths and ranges

- `l_price_ptr` (read, length `n_elements`): `l_extendedprice`, one per `lineitem` row; `n_elements` is `l_extendedprice.numel()`.
- `keep_mask_ptr` (read, length `n_elements`): `keep_mask`, a bool per `lineitem` row.
- `out_ptr` (readwrite, length `1`): `torch.zeros(1)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements` (size): `l_extendedprice.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 1. The plan launches only when a `lineitem` row is kept, so the table has rows.
