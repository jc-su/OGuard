# q07_core_triton__q7_kernel_7

`triton.py` is the kernel `_q7_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q07/round_05.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q7_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The `@triton.autotune` decorator is replaced by the values of this configuration, `BLOCK_SIZE = 8192`, 4 warps. The autotuner lists 16 configurations: `BLOCK_SIZE` 1024, 2048, 4096, 8192, `num_warps` 4 and 8, `num_stages` 2 and 4. The module states `CONSTANTS` and `NUM_WARPS` and no `num_stages`, and the kernel has no loop that Triton pipelines (the front end's PTX is the same for `num_stages` 2, 4, and Triton's default), so the two configurations that differ only in `num_stages` are one corpus kernel. The eight kernels are the configurations in the autotuner's order, the first without a suffix and the others `_2` to `_8`; this is number 7.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_launch_q7`, called once by `_query_core`, launches `_q7_kernel[(cdiv(n, BLOCK_SIZE),)]` through the autotuner.

## Lengths and ranges

- `supp_ptr` (read, length `ker_n_elements`): the 0/1 nation code `(joined["supp_nation"] == nation_b).astype("int32")` of each joined row; the kernel keeps only the values 0 and 1.
- `cust_ptr` (read, length `ker_n_elements`): the 0/1 nation code `(joined["cust_nation"] == nation_b).astype("int32")` of each joined row.
- `shipdate_ptr` (read, length `ker_n_elements`): `l_shipdate` in days, one per joined row; the kernel bounds it by `date_lo` and `date_hi` only, not the year it computes.
- `price_ptr` (read, length `ker_n_elements`): a column of the joined frame, one per row; `ker_n_elements` is `price.numel()`.
- `disc_ptr` (read, length `ker_n_elements`): a column of the joined frame, one per row.
- `date_lo` (scalar): from the query parameter; no range.
- `date_hi` (scalar): from the query parameter; no range.
- `out_ptr` (readwrite, length `8`): `torch.zeros(8)`; the kernel adds to it with `tl.atomic_add`.
- `ker_n_elements` (size): `price.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `ker_n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
