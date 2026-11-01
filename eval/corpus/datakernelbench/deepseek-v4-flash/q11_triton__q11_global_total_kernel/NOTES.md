# q11_triton__q11_global_total_kernel

`triton.py` is the kernel `_q11_global_total_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q11/round_04.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q11_global_total_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 4096`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_query_core` launches `_q11_global_total_kernel[(cdiv(n, 4096),)]` once with `BLOCK_SIZE=4096` and `num_warps=4`; the kernel also declares `num_warps` as a `tl.constexpr` argument, which the same keyword sets to 4.

## Lengths and ranges

- `supplycost_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `ps_partkey.numel()`.
- `availqty_ptr` (read, length `n_elements`): a column of the joined frame, one per row; `n_elements` is `ps_partkey.numel()`.
- `total_out` (readwrite, length `1`): `torch.zeros(1)`; each program adds to it with `tl.atomic_add`.
- `n_elements` (size): `ps_partkey.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
