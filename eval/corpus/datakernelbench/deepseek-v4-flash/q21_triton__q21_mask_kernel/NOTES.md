# q21_triton__q21_mask_kernel

`triton.py` is the kernel `_q21_mask_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q21/round_03.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q21_mask_kernel` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 4096`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`run_query` launches `_q21_mask_kernel[(cdiv(n, 4096),)]` once with `BLOCK_SIZE=4096` and Triton's default of 4 warps.

## Lengths and ranges

- `l1_orderkey_ptr` (read, length `n_elements`): `l_orderkey` of each row of `l1`; `n_elements` is `l1_orderkey_t.numel()`.
- `l1_late_ptr` (read, length `n_elements`): `l1["late"]`, a bool tensor.
- `supp_ids_ptr` (read, length `n_elements`): `supp_id` of each row of `l1`, in `[0, num_suppliers)`; the kernel only compares it, so no index role.
- `lo_orderkeys_ptr` (read, length `n_lo`): `agg["l_orderkey"]`, sorted; the kernel does not access it.
- `lo_distinct_supp_ptr` (read, length `n_lo`): a column of `agg`, one per order.
- `lo_late_distinct_supp_ptr` (read, length `n_lo`): a column of `agg`, one per order.
- `precomputed_idx_ptr` (offsets `lo_distinct_supp_ptr`, length `n_elements`): `torch.searchsorted(lo_orderkeys_t, l1_orderkey_t)`, whose values lie in `[0, n_lo]`, the range of offsets into `lo_distinct_supp_ptr` (and `lo_late_distinct_supp_ptr`, of the same length).
- `mask_out_ptr` (write, length `n_elements`): `torch.zeros(n, int32)`.
- `n_elements` (size): `l1_orderkey_t.numel()`, below 2^31 at scale factor 10.
- `num_suppliers` (size): `len(supp_n)`.

Least values of the size symbols: `n_elements` 0, `n_lo` 0, `num_suppliers` 0. The plan launches on whatever rows its joins and filters give, which can be none.
