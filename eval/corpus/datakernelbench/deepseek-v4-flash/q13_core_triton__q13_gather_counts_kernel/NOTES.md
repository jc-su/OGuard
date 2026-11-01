# q13_core_triton__q13_gather_counts_kernel

`triton.py` is the kernel `_q13_gather_counts_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q13/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q13_gather_counts_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 1024`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code. The plan's other kernel (`_q13_scatter_add_kernel`) is left out.
- The kernel's body and parameters are the plan's.

## Launch

`_launch_gather`, called once by `_query_core`, launches `_q13_gather_counts_kernel[(cdiv(n, 1024),)]` with `BLOCK_SIZE=1024` and Triton's default of 4 warps.

## Lengths and ranges

- `c_custkey_ptr` (read, length `n_elements`): `c_custkey`, one per customer; `n_elements` is `c_custkey.numel()`. The largest value is `num_customers`, but the plan does not check that the keys are non-negative, so no index role.
- `order_counts_ptr` (read, length `num_customers + 1`): `order_counts`, `torch.zeros(num_customers + 1)` with `num_customers = max(c_custkey)`, filled by `_q13_scatter_add_kernel`.
- `c_count_out_ptr` (write, length `n_elements`): `torch.zeros(n, int64)`.
- `n_elements` (size): `c_custkey.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0, `num_customers` 0. The plan launches on the whole `customer` table, which can be empty; `num_customers` is its largest key.
