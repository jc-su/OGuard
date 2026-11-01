# q21_core_triton__q21_filter_kernel

`triton.py` is the kernel `_q21_filter_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q21/round_06.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q21_filter_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 1024`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_query_core` launches `_q21_filter_kernel[(cdiv(n_l1, 1024),)]` once, when `l1` has rows, with `BLOCK_SIZE=1024` and Triton's default of 4 warps.

## Lengths and ranges

- `l1_orderkey_ptr` (read, length `n_l1`): `l_orderkey` of each row of `l1`; `n_l1` is `l1_orderkey.numel()`.
- `l1_late_ptr` (read, length `n_l1`): `l1["late"]` as `int8`, one per row.
- `supp_ids_ptr` (read, length `n_l1`): `supp_id` of each row of `l1`; the kernel only compares and copies it.
- `lo_orderkeys_ptr` (read, length `n_lo`): `agg["l_orderkey"]`, sorted; `n_lo` is `lo_orderkeys.numel()`.
- `lo_distinct_supp_ptr` (read, length `n_lo`): a column of `agg`, one per order.
- `lo_late_distinct_supp_ptr` (read, length `n_lo`): a column of `agg`, one per order.
- `out_valid_ptr` (write, length `n_l1`): `torch.full((n_l1,), -1, int64)`.
- `out_count_ptr` (write, length `n_l1`): `torch.zeros(n_l1, int64)`.
- `n_l1` (size): `l1_orderkey.numel()`, below 2^31 at scale factor 10; the plan launches only when it is positive.
- `n_lo` (size): `lo_orderkeys.numel()`, below 2^31 at scale factor 10; `agg` aggregates the rows that `l1` is drawn from, so it has an order whenever `l1` has a row.

Least values of the size symbols: `n_l1` 1, `n_lo` 1. The plan launches only when `l1` has rows, and `agg` then has at least one order.
