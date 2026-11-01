# q22_core_triton__q22_avg_kernel

`triton.py` is the kernel `_q22_avg_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q22/round_05.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q22_avg_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 4096`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code. The plan's other kernel (`_q22_filter_group_kernel`) is left out.
- The kernel's body and parameters are the plan's.

## Launch

`_query_core` launches `_q22_avg_kernel[(cdiv(n, 4096),)]` once with `BLOCK_SIZE=4096` and Triton's default of 4 warps, then `_q22_filter_group_kernel`.

## Lengths and ranges

- `acctbal_ptr` (read, length `n_elements`): `c_acctbal`, one per customer; `n_elements` is `c_custkey.numel()`.
- `in_mask_ptr` (read, length `n_elements`): `in_cntrycode_mask`, a bool per customer.
- `avg_out_ptr` (readwrite, length `2`): `torch.zeros(2, float64)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements` (size): `c_custkey.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on the whole `customer` table, which can be empty.
