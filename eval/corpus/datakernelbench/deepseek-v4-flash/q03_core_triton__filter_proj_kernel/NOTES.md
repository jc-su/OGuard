# q03_core_triton__filter_proj_kernel

`triton.py` is the kernel `_filter_proj_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q03/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_filter_proj_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 4096`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_query_core` launches `_filter_proj_kernel[(cdiv(n, 4096),)]` once with `BLOCK_SIZE=4096` and Triton's default of 4 warps.

## Lengths and ranges

- `l_orderkey_ptr` (readwrite, length `n_elements`): `l_orderkey.clone()` of the joined frame, one per row, which the kernel loads and stores back; `n_elements` is `l_orderkey.shape[0]`.
- `l_extendedprice_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `l_discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `l_shipdate_ptr` (read, length `n_elements`): `l_shipdate` in days, one per joined row.
- `o_orderdate_ptr` (readwrite, length `n_elements`): `o_orderdate_days.clone()`, which the kernel loads and stores back.
- `o_shippriority_ptr` (readwrite, length `n_elements`): `o_shippriority.clone()`, which the kernel loads and stores back.
- `rev_out_ptr` (write, length `n_elements`): `torch.zeros(n, float64)`.
- `shipdate_lo` (scalar): from the query parameter; no range.
- `orderdate_hi` (scalar): from the query parameter; no range.
- `n_elements` (size): `l_orderkey.shape[0]`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
