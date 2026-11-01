# q13_triton__count_orders_kernel

Source: `azure__gpt-5.5/triton/full/queries/q13/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_count_orders_kernel` of `_KERNEL_SOURCE`
(plan lines 18 to 35).

## Extraction edits

- Renamed `_count_orders_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan lines 183 to 192) launches the kernel when
`n_orders > 0`, with grid `(triton.cdiv(n_orders, 1024),)`,
`BLOCK_SIZE=count_block_size` (1024), and no `num_warps` (4). `n_orders` and
`counts_len` are `tl.constexpr` arguments whose values are table sizes:
`n_orders = o_custkey.numel()`, 15,000,000 at scale factor 10, and
`counts_len = num_customers + 1` with `num_customers = max(c_custkey)`,
1,500,001 at scale factor 10.

## Lengths and ranges

With the sizes fixed by the constants, every length is a constant.

- `o_custkey_ptr`: `o_custkey` of `orders`, int64, 15,000,000 elements.
- `excluded_ptr`: the `o_comment` pattern match of `orders` as a
  `torch.bool` tensor (`*i1`), 15,000,000 elements.
- `counts_ptr`: `torch.zeros(counts_len)` int32, 1,500,001 elements, which the
  kernel adds to with `tl.atomic_add` (`readwrite`).
- The plan does not bound `o_custkey` by `counts_len`; the kernel checks
  `0 <= cust < counts_len` before the atomic.
