# q13_core_triton__hist_customer_counts_i32_kernel

Source: `azure__gpt-5.5/triton/core/queries/q13/round_06.py` (`output_round` 6),
copied as `plan.py`. The kernel is `_hist_customer_counts_i32_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_hist_customer_counts_i32_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, when `n_customer_rows > 0` and
`n_customer_rows != num_customers`, with grid `(ceil(n_customer_rows / 1024),)`,
`BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `c_custkey_ptr`: the customer keys, `n_customers = c_custkey.numel()` rows (`i32`). `counts_n = max(c_custkey) + 1` is sized from the maximum of the same keys, and TPC-H keys are positive, so `c_custkey_ptr` is `index-in` `counts_ptr`.
- `counts_ptr`: the `counts_n` order counts; read only. `counts_n` is a size symbol that the kernel does not receive, least value 1.
- `hist_ptr`: the 64 `int32` bins, added to with `tl.atomic_add` (`readwrite`).
- `n_customers`: size symbol with least value 1.

## Branch

The plan takes this branch only when the customer row count differs from the
largest customer key, which does not hold at scale factor 10 (1,500,000 rows,
keys 1 to 1,500,000).
