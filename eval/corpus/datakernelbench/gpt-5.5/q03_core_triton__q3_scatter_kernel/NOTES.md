# q03_core_triton__q3_scatter_kernel

Source: `azure__gpt-5.5/triton/core/queries/q03/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q3_scatter_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q3_scatter_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_compact_q3_rows` launches the kernel once, when `n > 0` and `total > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `l_orderkey_ptr` (`int64`), `l_extendedprice_ptr`, `l_discount_ptr` (`float64`), `o_orderdate_days_ptr`, `o_shippriority_ptr` (`int32`): the joined columns, `n_elements` rows each (below 2^31, so `i32`).
- `keep_ptr`: the `n_elements` flags of `_q3_count_kernel`; read only.
- `block_offsets_ptr`: `torch.cumsum(counts) - counts`, `(n_elements + 1023) / 1024` elements; the quotient gives the buffer its own size symbol with least value 1. Read only.
- `out_lk_ptr` (`int64`), `out_rev_ptr` (`float64`), `out_od_ptr`, `out_sp_ptr` (`int32`): `torch.empty((total,))` each, `total = counts.sum()` elements, the number of kept rows; `total` is a size symbol that the kernel does not receive, least value 1 at this launch. Written only.
- `n_elements`: size symbol with least value 1.
