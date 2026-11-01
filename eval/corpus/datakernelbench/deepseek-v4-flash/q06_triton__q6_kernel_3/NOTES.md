# q06_triton__q6_kernel_3

`triton.py` is the kernel `_q6_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q06/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q6_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The `@triton.autotune` decorator is replaced by the values of this configuration, `BLOCK_SIZE = 2048`, 4 warps. The autotuner lists 16 configurations: `BLOCK_SIZE` 1024, 2048, 4096, 8192, `num_warps` 4 and 8, `num_stages` 2 and 4. The module states `CONSTANTS` and `NUM_WARPS` and no `num_stages`, and the kernel has no loop that Triton pipelines (the front end's PTX is the same for `num_stages` 2, 4, and Triton's default), so the two configurations that differ only in `num_stages` are one corpus kernel. The eight kernels are the configurations in the autotuner's order, the first without a suffix and the others `_2` to `_8`; this is number 3.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`run_query` launches `_q6_kernel[(cdiv(n, BLOCK_SIZE),)]` once, through the autotuner.

## Lengths and ranges

- `price_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `price.numel()`.
- `discount_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `price.numel()`.
- `quantity_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `price.numel()`.
- `shipdate_ptr` (read, length `n_elements`): `l_shipdate` in days, one per row.
- `date_lo` (scalar): from the query parameter; no range.
- `date_hi` (scalar): from the query parameter; no range.
- `thresh_ptr` (read, length `3`): `torch.tensor([discount_lo, discount_hi, quantity_hi])`.
- `out_ptr` (readwrite, length `1`): `torch.zeros(1)`; each program adds to it with `tl.atomic_add`.
- `n_elements` (size): `price.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
