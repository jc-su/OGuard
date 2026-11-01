# q09_triton__compute_amount_kernel

`triton.py` is the kernel `_compute_amount_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q09/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_compute_amount_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 4096`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_query_core` launches `_compute_amount_kernel[(cdiv(n, 4096),)]` once with `BLOCK_SIZE=4096` and Triton's default of 4 warps.

## Lengths and ranges

- `price_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `l_extendedprice.numel()`.
- `discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `l_extendedprice.numel()`.
- `cost_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `l_extendedprice.numel()`.
- `qty_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `l_extendedprice.numel()`.
- `out_ptr` (write, length `n_elements`): `torch.empty(n)`.
- `n_elements` (size): `l_extendedprice.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
