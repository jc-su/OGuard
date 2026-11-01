# q13_core_triton__q13_scatter_add_kernel

`triton.py` is the kernel `_q13_scatter_add_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q13/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q13_scatter_add_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 1024`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code. The plan's other kernel (`_q13_gather_counts_kernel`) is left out.
- The kernel's body and parameters are the plan's.

## Launch

`_launch_scatter_add`, called once by `_query_core`, launches `_q13_scatter_add_kernel[(cdiv(n, 1024),)]` with `BLOCK_SIZE=1024` and Triton's default of 4 warps.

## Lengths and ranges

- `o_custkey_ptr` (read, length `n_orders`): `o_custkey` of the orders whose comment does not match; `n_orders` is `o_custkey.numel()`. The kernel bounds it below only, so no index role.
- `order_counts_ptr` (readwrite, length `num_customers + 1`): `torch.zeros(num_customers + 1, int64)` with `num_customers = max(c_custkey)`; the kernel adds to it with `tl.atomic_add`.
- `n_orders` (size): `o_custkey.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_orders` 0, `num_customers` 0. The plan launches on whatever orders its filter keeps, which can be none; `num_customers` is the largest customer key.
