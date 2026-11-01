# q19_triton__q19_kernel

`triton.py` is the kernel `_q19_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q19/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q19_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The `@triton.autotune` decorator is replaced by the values of this configuration, `BLOCK_SIZE = 1024`, 4 warps. The autotuner lists 16 configurations: `BLOCK_SIZE` 1024, 2048, 4096, 8192, `num_warps` 4 and 8, `num_stages` 2 and 4. The module states `CONSTANTS` and `NUM_WARPS` and no `num_stages`, and the kernel has no loop that Triton pipelines (the front end's PTX is the same for `num_stages` 2, 4, and Triton's default), so the two configurations that differ only in `num_stages` are one corpus kernel. The eight kernels are the configurations in the autotuner's order, the first without a suffix and the others `_2` to `_8`; this is number 1.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`run_query` launches `_q19_kernel[(cdiv(n, BLOCK_SIZE),)]` once, through the autotuner.

## Lengths and ranges

- `l_quantity_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `l_quantity.numel()`.
- `l_extendedprice_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `l_quantity.numel()`.
- `l_discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `l_quantity.numel()`.
- `p_brand_id_ptr` (read, length `n_elements`): a code or `p_size` of each joined row.
- `p_container_grp_ptr` (read, length `n_elements`): a code or `p_size` of each joined row.
- `p_size_ptr` (read, length `n_elements`): a code or `p_size` of each joined row.
- `brand12_id` (scalar): the constant 12 of `run_query`.
- `brand23_id` (scalar): the constant 23.
- `brand34_id` (scalar): the constant 34.
- `sm_grp` (scalar): the constant 1.
- `md_grp` (scalar): the constant 2.
- `lg_grp` (scalar): the constant 3.
- `thresh1_lo` (scalar): the Python float 1.0, which Triton passes as `fp32`.
- `thresh1_hi` (scalar): the Python float 11.0.
- `thresh2_lo` (scalar): the Python float 10.0.
- `thresh2_hi` (scalar): the Python float 20.0.
- `thresh3_lo` (scalar): the Python float 20.0.
- `thresh3_hi` (scalar): the Python float 30.0.
- `out_ptr` (readwrite, length `1`): `torch.zeros(1)`; each program adds to it with `tl.atomic_add`.
- `n_elements` (size): `l_quantity.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
