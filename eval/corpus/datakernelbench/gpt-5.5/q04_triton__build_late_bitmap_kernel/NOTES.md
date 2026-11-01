# q04_triton__build_late_bitmap_kernel

Source: `azure__gpt-5.5/triton/full/queries/q04/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_build_late_bitmap_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_build_late_bitmap_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_launch_build_late_bitmap` (plan.py line 124) launches the kernel once with grid
`(triton.cdiv(n_elements, 1024),)`, `BLOCK_SIZE=1024` (`block_size=1024` of
`run_query`), and `num_warps=8`.

## Lengths and ranges

- `l_orderkey_ptr` (int64), `l_commitdate_days_ptr`, `l_receiptdate_days_ptr`
  (int32 epoch days): lineitem columns, `n_elements = l_orderkey.numel()` rows.
- `bitmap_ptr`: `torch.zeros((max(bitmap_size, 1),), dtype=torch.uint8)`; the
  launch happens only when `bitmap_size > 0`, so it has `bitmap_size` elements,
  which the kernel only stores into (`write`).
- `bitmap_size`: `int(df_lineitem["l_orderkey"].max()) + 1`, computed from the same
  `l_orderkey` values, which are positive, so every key lies in
  `[0, bitmap_size - 1]`: `l_orderkey_ptr` is `index-in` `bitmap_ptr`.
  60,000,001 at scale factor 10, so `i32`.
- `n_elements`, `bitmap_size`: size symbols with least value 1; the launcher
  returns before the launch when `n_elements == 0` or `bitmap_size <= 0`.
