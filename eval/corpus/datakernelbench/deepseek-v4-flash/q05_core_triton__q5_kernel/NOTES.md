# q05_core_triton__q5_kernel

`triton.py` is the kernel `_q5_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q05/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q5_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The `@triton.autotune` decorator is replaced by the values of this configuration, `BLOCK_SIZE = 1024`, 4 warps. The autotuner lists 24 configurations: `BLOCK_SIZE` 1024, 2048, 4096, 8192, `num_warps` 4 and 8, `num_stages` 2, 3, and 4. The module states `CONSTANTS` and `NUM_WARPS` and no `num_stages`, and the kernel has no loop that Triton pipelines (the front end's PTX is the same for `num_stages` 2, 3, 4, and Triton's default), so the three configurations that differ only in `num_stages` are one corpus kernel. The eight kernels are the configurations in the autotuner's order, the first without a suffix and the others `_2` to `_8`; this is number 1.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code. The `tl.constexpr` `num_nations` takes the value 5, `len(nations_unique)` for the plan's region ASIA: the TPC-H nation table has five nations per region at every scale factor.
- The kernel's body and parameters are the plan's.

## Launch

`_query_core` launches `_q5_kernel[(cdiv(n, BLOCK_SIZE),)]` once, through the autotuner, with `num_nations = 5`.

## Lengths and ranges

- `nation_id_ptr` (read, length `n_elements`): `final_df["nation_id"]`, the row number of `nations_unique` joined by `n_nationkey`; the kernel only compares it, so no index role.
- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `nation_id.numel()`.
- `discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `out_ptr` (readwrite, length `5`): `torch.zeros(num_nations)` with `num_nations = len(nations_unique)`, 5 for the plan's region ASIA, the value of the `tl.constexpr` `num_nations`; the kernel adds to it with `tl.atomic_add`.
- `n_elements` (size): `nation_id.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
