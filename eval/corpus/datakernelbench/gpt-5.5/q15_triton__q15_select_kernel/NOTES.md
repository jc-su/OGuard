# q15_triton__q15_select_kernel

Source: `azure__gpt-5.5/triton/full/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q15_select_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q15_select_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` launches the kernel once with grid `(triton.cdiv(n_sup, 1024),)`,
`n_sup` as a `tl.constexpr`, `BLOCK_SIZE=1024`, and `num_warps=8`. `n_sup` is
the rows of `supplier`, 100,000 at scale factor 10.

## Lengths and ranges

- `totals_ptr` (`float64`), `present_ptr` (`uint8`), `sorted_suppkey_ptr`
  (`int64`): 100000 elements each, read.
- `max_ptr`: `torch.empty(1)`, one `float64` that the final-max kernel wrote,
  read.
- `count_ptr`: `torch.empty(1)` of `int32`, zeroed by the plan's init kernel;
  the kernel adds to it with `tl.atomic_add` and uses the old value as the base
  of its output positions (`readwrite`).
- `sel_key_ptr` (`int64`), `sel_rev_ptr` (`float64`): `torch.empty(n_sup)`,
  100000 elements each, written at `base + prefix`.
- The lengths are constants because `n_sup` is a constexpr of this
  configuration.
