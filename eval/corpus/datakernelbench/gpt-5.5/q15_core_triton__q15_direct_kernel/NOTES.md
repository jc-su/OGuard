# q15_core_triton__q15_direct_kernel

Source: `azure__gpt-5.5/triton/core/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_q15_direct_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_q15_direct_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_aggregate_direct_contiguous` launches the kernel once, when `n > 0` and
`n_sup > 0`, with grid `(triton.cdiv(n, 1024),)`, `BLOCK_SIZE=1024`, and
`num_warps=8`.

## Lengths and ranges

- `l_suppkey_ptr` (`int64`), `l_extendedprice_ptr`, `l_discount_ptr` (`float64`), `l_shipdate_days_ptr` (`int32`): the lineitem columns, `n_elements = l_suppkey.numel()` rows each (59,986,052 at scale factor 10, so `i32`); `l_suppkey_ptr` is offset by 1 and bounded by the kernel, so `read`.
- `totals_ptr`: `torch.zeros(n_sup, dtype=torch.float64)`, added to with `tl.atomic_add` (`readwrite`); `present_ptr`: `torch.zeros(n_sup, dtype=torch.int8)`, written only.
- `n_elements`, `n_suppliers` (`s_suppkey.numel()`, 100,000): size symbols with least value 1; the launch runs only when both are positive.
- `shipdate_lo_days`, `shipdate_hi_days`: `i32` comparison thresholds only.

## Branch

The plan takes this branch when `s_suppkey` equals `arange(1, n_sup + 1)`, which
holds for the benchmark's supplier table at scale factor 10.
