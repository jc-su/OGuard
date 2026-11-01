# q01_core_triton__q1_filter_and_reduce

`triton.py` is the kernel `_q1_filter_and_reduce` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/core/queries/q01/round_09.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q1_filter_and_reduce` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The plan launches the kernel with `BLOCK_SIZE = 4096`, 4 warps; `CONSTANTS` and `NUM_WARPS` state them.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_launch_q1`, called once by `_query_core`, launches `_q1_filter_and_reduce[(cdiv(n, 4096),)]` with `BLOCK_SIZE=4096` and Triton's default of 4 warps.

## Lengths and ranges

- `returnflag_ptr` (read, length `n_elements`): the category code of `l_returnflag`, one per row; the kernel only compares the group it computes from it.
- `linestatus_ptr` (read, length `n_elements`): the category code of `l_linestatus`, one per row.
- `quantity_ptr` (read, length `n_elements`): a `lineitem` column, one per row; `n_elements` is `l_quantity.numel()`.
- `extendedprice_ptr` (read, length `n_elements`): a `lineitem` column, one per row.
- `discount_ptr` (read, length `n_elements`): a `lineitem` column, one per row.
- `tax_ptr` (read, length `n_elements`): a `lineitem` column, one per row.
- `shipdate_ptr` (read, length `n_elements`): `l_shipdate` in days, one per row.
- `shipdate_hi` (scalar): from the query parameter; no range.
- `out_qty` (readwrite, length `n_groups`): `torch.zeros(n_groups, float64)`; the kernel adds to it with `tl.atomic_add`.
- `out_price` (readwrite, length `n_groups`): `torch.zeros(n_groups, float64)`; the kernel adds to it with `tl.atomic_add`.
- `out_disc_price` (readwrite, length `n_groups`): `torch.zeros(n_groups, float64)`; the kernel adds to it with `tl.atomic_add`.
- `out_charge` (readwrite, length `n_groups`): `torch.zeros(n_groups, float64)`; the kernel adds to it with `tl.atomic_add`.
- `out_disc` (readwrite, length `n_groups`): `torch.zeros(n_groups, float64)`; the kernel adds to it with `tl.atomic_add`.
- `out_count` (readwrite, length `n_groups`): `torch.zeros(n_groups, float64)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements` (size): `l_quantity.numel()`, below 2^31 at scale factor 10.
- `n_groups` (size): `n_rf * n_ls`, the length of the six outputs, with `n_rf` and `n_ls` the largest code plus one (1 for an empty column).
- `n_ls` (scalar): `n_ls`, the largest line status code plus one, or 1 for an empty column; the kernel only multiplies by it.

Least values of the size symbols: `n_elements` 0, `n_groups` 1. The plan launches on the whole `lineitem` table, which can be empty; `n_groups` is a product of two counts of at least 1.
