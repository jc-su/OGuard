# q21_core_triton__q21_core_kernel

Source: `azure__gpt-5.5/triton/core/queries/q21/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q21_core_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q21_core_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_q21_core` launches the kernel once, when `n_l1 > 0` and
`num_suppliers > 0`, with grid `(triton.cdiv(n_l1, 256),)`, `BLOCK_SIZE=256`,
`SEARCH_ITERS=bit_length(lo_n)`, and `num_warps=4`. `lo_n` is the number of
distinct order keys of the lineitems of orders with status `F`, one per such order;
TPC-H gives status `F` to about 48.6% of the 15,000,000 orders at scale factor 10
(729,413 of 1,500,000 at scale factor 1), so `lo_n` lies between 2^22 and 2^23 and
`SEARCH_ITERS` is 23.

## Lengths and ranges

- `l1_orderkey_ptr` (`int64`), `l1_late_ptr` (a `torch.bool` tensor, `*i1`), `supp_ids_ptr` (`int64`): the `l1` rows, `n_l1` each (below 2^31, so `i32`).
- `supp_ids_ptr`: `supp_id = range(len(supp_n))` by construction and `num_suppliers = len(supp_n)`, so every value lies in `[0, num_suppliers)`: `index-in` `counts_ptr`.
- `lo_orderkeys_ptr`, `lo_distinct_supp_ptr`, `lo_late_distinct_supp_ptr`: the per-order aggregate, `lo_n` rows each (`i32`); `lo_orderkeys_ptr` is sorted, which the review does not rely on.
- `counts_ptr`: `torch.zeros(num_suppliers, dtype=torch.int64)`, added to with `tl.atomic_add` (`readwrite`); `num_suppliers` is a size symbol that the kernel does not receive, least value 1.
- `n_l1`, `lo_n`: size symbols with least value 1 (`lo_n` is positive whenever `n_l1` is, since every `l1` row's order is in the aggregate; the review uses only `lo_n >= 1`).
