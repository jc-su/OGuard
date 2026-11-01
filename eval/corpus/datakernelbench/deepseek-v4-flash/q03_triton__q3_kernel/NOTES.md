# q03_triton__q3_kernel

`triton.py` is the kernel `_q3_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q03/round_07.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q3_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 16384`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`run_query` launches `_q3_kernel[(cdiv(n, 16384),)]` once with `BLOCK_SIZE=16384` and Triton's default of 4 warps.

## Lengths and ranges

- `l_orderkey_ptr` (read, length `n_elements`): `l_orderkey` of each joined row; `n_elements` is `l_orderkey.numel()`.
- `l_extendedprice_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `l_discount_ptr` (read, length `n_elements`): a column of the joined frame, one per row.
- `l_shipdate_days_ptr` (read, length `n_elements`): an int32 column of the joined frame, one per row.
- `o_orderdate_days_ptr` (read, length `n_elements`): an int32 column of the joined frame, one per row.
- `o_shippriority_ptr` (read, length `n_elements`): an int32 column of the joined frame, one per row.
- `shipdate_lo_days` (scalar): from the query parameter; no range.
- `orderdate_hi_days` (scalar): from the query parameter; no range.
- `out_rev_ptr` (write, length `n_elements`): `torch.empty(n, float64)`.
- `out_lk_ptr` (write, length `n_elements`): `torch.empty(n, int64)`.
- `out_od_ptr` (write, length `n_elements`): `torch.empty(n, int32)`.
- `out_sp_ptr` (write, length `n_elements`): `torch.empty(n, int32)`.
- `n_elements` (size): `l_orderkey.numel()`, below 2^31 at scale factor 10.

Least values of the size symbols: `n_elements` 0. The plan launches on whatever rows its joins and filters give, which can be none.
