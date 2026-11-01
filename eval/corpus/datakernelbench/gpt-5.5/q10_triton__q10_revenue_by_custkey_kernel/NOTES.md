# q10_triton__q10_revenue_by_custkey_kernel

Source: `azure__gpt-5.5/triton/full/queries/q10/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q10_revenue_by_custkey_kernel` of
`_KERNEL_SOURCE` (plan lines 19 to 40).

## Extraction edits

- Renamed `_q10_revenue_by_custkey_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_revenue_by_custkey` (plan lines 67 to 98) launches the kernel once
with grid `(triton.cdiv(n_rows, BLOCK_SIZE),)`, `BLOCK_SIZE=block_size` (the
default 256 of `run_query`), and `num_warps=4`.

## Lengths and ranges

- `custkey_ptr` (int64), `price_ptr`, `discount_ptr` (float64): the columns
  `o_custkey`, `l_extendedprice`, `l_discount` of the lineitem and orders join
  `joined`, `n_rows` rows each; `n_rows = l_extendedprice.numel()`, `i32`.
- `n_rows`: size symbol with least value 1; the launcher returns before the
  launch when `n_rows == 0`.
- `revenue_ptr`: `torch.zeros(n_revenue)` float64, which the kernel adds to
  with `tl.atomic_add` (`readwrite`).
- `n_revenue`: `max(c_custkey) + 1` (1,500,001 at scale factor 10), `i32`, a
  size symbol with least value 1; the launcher returns when `n_revenue <= 0`.
  The plan does not bound `o_custkey` by it; the kernel checks
  `0 <= custkey < n_revenue` before the atomic.
