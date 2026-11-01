# q13_triton__hist_partial_kernel

Source: `azure__gpt-5.5/triton/full/queries/q13/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_hist_partial_kernel` of `_KERNEL_SOURCE`
(plan lines 38 to 64).

## Extraction edits

- Renamed `_hist_partial_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan lines 194 to 213) launches the kernel once with grid
`(num_customer_blocks,)`, `num_customer_blocks = triton.cdiv(n_customers, 1024)`,
`SMALL_BINS=small_bins` (256), `BLOCK_SIZE=customer_block_size` (1024), and
no `num_warps` (4). `n_customers = c_custkey.numel()` (1,500,000 at scale
factor 10) and `counts_len = max(c_custkey) + 1` (1,500,001) are
`tl.constexpr` arguments; `num_customer_blocks` is 1,465.

## Lengths and ranges

With the sizes fixed by the constants, every length is a constant.

- `c_custkey_ptr`: `c_custkey` of `customer`, int64, 1,500,000 elements.
- `counts_ptr`: the int32 per-customer counts of `_count_orders_kernel`,
  1,500,001 elements; read only.
- `partial_hist_ptr`: `torch.zeros((num_customer_blocks, small_bins))` int64,
  1,465 * 256 = 375,040 elements, added to with `tl.atomic_add` (`readwrite`).
- `overflow_ptr`: `torch.zeros(1)` int64, added to with `tl.atomic_add`
  (`readwrite`).
- `run_query` returns before `_query_core` for an empty `customer`, and
  `_query_core` returns when `n_customers == 0`.
