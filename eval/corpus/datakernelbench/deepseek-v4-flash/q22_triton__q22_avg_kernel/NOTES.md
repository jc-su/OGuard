# q22_triton__q22_avg_kernel

`triton.py` is the kernel `_q22_avg_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q22/round_07.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q22_avg_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 4096`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code and the plan's other kernels (`_q22_filter_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`run_query` launches `_q22_avg_kernel[(cdiv(n_customer, 4096),)]` once with `BLOCK_SIZE=4096` and Triton's default of 4 warps.

## Lengths and ranges

- `acctbal_ptr` (read, length `n_elements`): `c_acctbal`, one per customer; `n_elements` is `c_acctbal.numel()`.
- `cntrycode_ptr` (read, length `n_elements`): the country code of each customer's phone.
- `in_mask_ptr` (read, length `n_elements`): `in_mask_series` as int32, one per customer.
- `avg_out_ptr` (readwrite, length `1`): `torch.zeros(1)`; the kernel adds to it with `tl.atomic_add`.
- `count_out_ptr` (readwrite, length `1`): `torch.zeros(1, int64)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements` (size): `c_acctbal.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
