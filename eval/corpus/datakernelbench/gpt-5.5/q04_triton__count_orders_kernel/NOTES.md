# q04_triton__count_orders_kernel

Source: `azure__gpt-5.5/triton/full/queries/q04/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_count_orders_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_count_orders_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_count_orders` (plan.py line 160) launches the kernel once with grid
`(triton.cdiv(n_elements, 1024),)`, `BLOCK_SIZE=1024`, `MAX_PRIORITIES` equal to
`num_priorities`, the category count of `o_orderpriority`, and `num_warps=8`.
At scale factor 10 there are 5 priorities (`1-URGENT` to `5-LOW`, as the
benchmark's summary of orders at sf10 states), so `MAX_PRIORITIES=5`.

## Lengths and ranges

- `o_orderkey_ptr` (int64), `o_orderdate_days_ptr` (int32),
  `o_priority_codes_ptr` (int32 category codes): orders columns,
  `n_elements = o_orderkey.numel()` rows (15,000,000 at scale factor 10).
- `bitmap_ptr`: the uint8 bitmap that `_build_late_bitmap_kernel` filled, of
  `bitmap_size = late_bitmap.numel()` elements. `o_orderkey` is not bounded by
  the plan against this length (the bitmap is sized from lineitem's keys), so it
  is `read`; the kernel bounds each key by `bitmap_size`.
- `counts_ptr`: `torch.zeros((num_priorities,), dtype=torch.int64)`, 5 elements,
  which the kernel adds to with `tl.atomic_add` (`readwrite`).
- `n_elements`: size symbol with least value 1 (the launcher returns when it is 0);
  `bitmap_size`: `max(bitmap_size, 1)`, least value 1.
- `date_lo_days`, `date_hi_days`: Python `int` epoch days of 1993-07-01 and
  1993-10-01, `i32`; comparison bounds only.
