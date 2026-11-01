# q15_core_triton__q15_hash_agg_kernel

Source: `azure__gpt-5.5/triton/core/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q15_hash_agg_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q15_hash_agg_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_aggregate_hash_general` launches the kernel once, when `n > 0`, with grid
`(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, `MAX_PROBES=128`,
`table_mask=table_size - 1`, and `num_warps=8`.

## Lengths and ranges

- `l_suppkey_ptr` (`int64`), `l_extendedprice_ptr`, `l_discount_ptr` (`float64`), `l_shipdate_days_ptr` (`int32`): the lineitem columns, `n_elements` rows each (`i32`); `l_suppkey_ptr` is hashed only, so `read`.
- `hash_keys_ptr`, `hash_vals_ptr`: the `table_mask + 1` entries that `_q15_hash_insert_kernel` filled; read only here.
- `totals_ptr` (`float64`), `present_ptr` (`int8`): `torch.zeros(n_sup)` each; `n_sup` is a size symbol that the kernel does not receive, least value 1.
- `n_elements`: size symbol with least value 1; `table_mask`: a power of two minus one of at least 15 (`i32`).
- `shipdate_lo_days`, `shipdate_hi_days`: `i32` comparison thresholds only.

## Branch

The plan takes the hash path only when `s_suppkey` differs from `arange(1, n_sup + 1)`,
which does not hold for the benchmark's supplier table at scale factor 10.
