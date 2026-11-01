# q13_core_triton__hist_dense_customer_counts_i32_kernel

Source: `azure__gpt-5.5/triton/core/queries/q13/round_06.py` (`output_round` 6),
copied as `plan.py`. The kernel is `_hist_dense_customer_counts_i32_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_hist_dense_customer_counts_i32_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once, when `n_customer_rows > 0` and
`n_customer_rows == num_customers`, with grid `(ceil(n_customer_rows / 1024),)`,
`BLOCK_SIZE=1024`, and `num_warps=4`.

## Lengths and ranges

- `counts_ptr`: the `counts_n = num_customers + 1` order counts of `_scatter_order_counts_i32_kernel`. On this branch `num_customers == n_customer_rows`, so the plan establishes `n_customers + 1` elements; read only.
- `hist_ptr`: the 64 `int32` bins, added to with `tl.atomic_add` (`readwrite`).
- `n_customers`: `c_custkey.numel()`, size symbol with least value 1 (1,500,000 at scale factor 10, so `i32`).

## Branch

The plan takes this branch when the customer row count equals the largest
customer key, which holds at scale factor 10 (1,500,000 rows, keys 1 to
1,500,000).
