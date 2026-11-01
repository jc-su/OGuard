# q22_core_triton__q22_filter_group_kernel

`triton.py` is the kernel `_q22_filter_group_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q22/round_05.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q22_filter_group_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 4096`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code. The plan's other kernel (`_q22_avg_kernel`) is left out.
- The kernel's body and parameters are the plan's.

## Launch

`_query_core` launches `_q22_filter_group_kernel[(cdiv(n, 4096),)]` once with `BLOCK_SIZE=4096` and Triton's default of 4 warps, after `_q22_avg_kernel`.

## Lengths and ranges

- `custkey_ptr` (read, length `n_elements`): `c_custkey`, one per customer; `n_elements` is `c_custkey.numel()`.
- `acctbal_ptr` (read, length `n_elements`): `c_acctbal`, one per customer.
- `cntrycode_ptr` (read, length `n_elements`): the two-digit country code of each customer; the kernel only compares it.
- `in_mask_ptr` (read, length `n_elements`): `in_cntrycode_mask`, a bool per customer.
- `o_custkey_ptr` (read, length `o_custkey_n`): the distinct `o_custkey` values, sorted; `o_custkey_n` is `sorted_okeys.numel()`.
- `o_custkey_n` (size): `sorted_okeys.numel()`, below 2^31 at scale factor 10.
- `avg_val_ptr` (read, length `1`): `avg_val.unsqueeze(0)`, the average balance of `_q22_avg_kernel`.
- `out_numcust_ptr` (readwrite, length `7`): `torch.zeros(7, int64)`; the kernel adds to it with `tl.atomic_add`.
- `out_totacctbal_ptr` (readwrite, length `7`): `torch.zeros(7, float64)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements` (size): `c_custkey.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0, `o_custkey_n` 0. The plan launches on the whole `customer` table, which can be empty; the orders can be empty too.
