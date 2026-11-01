# q07_triton__q7_kernel_6

`triton.py` is the kernel `_q7_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q07/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q7_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The `@triton.autotune` decorator is replaced by the values of this configuration, `BLOCK_SIZE = 4096`, 8 warps. The autotuner lists 16 configurations: `BLOCK_SIZE` 1024, 2048, 4096, 8192, `num_warps` 4 and 8, `num_stages` 2 and 4. The module states `CONSTANTS` and `NUM_WARPS` and no `num_stages`, and the kernel has no loop that Triton pipelines (the front end's PTX is the same for `num_stages` 2, 4, and Triton's default), so the two configurations that differ only in `num_stages` are one corpus kernel. The eight kernels are the configurations in the autotuner's order, the first without a suffix and the others `_2` to `_8`; this is number 6.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_launch_q7` launches `_q7_kernel[(cdiv(n, BLOCK_SIZE),)]` once, through the autotuner.

## Lengths and ranges

- `supp_code_ptr` (read, length `n_elements`): the 0/1 nation code of each joined row; `n_elements` is `supp_code.numel()`.
- `cust_code_ptr` (read, length `n_elements`): the 0/1 nation code of each joined row; `n_elements` is `supp_code.numel()`.
- `shipdate_days_ptr` (read, length `n_elements`): `l_shipdate` in days, one per joined row.
- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `disc_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `n_elements` (size): `supp_code.numel()`, below 2^31 at scale factor 10.
- `shipdate_lo` (scalar): from the query parameter; no range.
- `shipdate_hi` (scalar): from the query parameter; no range.
- `out_revenue_ptr` (readwrite, length `8`): `torch.zeros(8)`; the kernel adds to it with `tl.atomic_add`.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
