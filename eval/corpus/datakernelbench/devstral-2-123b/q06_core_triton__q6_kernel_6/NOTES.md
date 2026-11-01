# q06_core_triton__q6_kernel_6

Kernel `_q6_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/core/queries/q06/round_03.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 3`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=4096`, `num_warps=8` (listed twice in the autotuner, with `num_stages=2` and `num_stages=4`).

## Extraction edits

- Renamed the kernel `_q6_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 4096}` and `NUM_WARPS = 8`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Dropped the decorator's `reset_to_zero` list (`out_ptr`), which the autotuner zeroes between its trial runs; the plan allocates these buffers zeroed before the launch.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`_launch_q6` launches `_q6_kernel[grid]` at plan lines 97-100 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 96); `_query_core` calls it at lines 114-117 and `run_query` at lines 151-161. The autotuner (lines 14-23) picks one of sixteen configurations, eight distinct in `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `price_ptr` (read, float64, length `n_elements`): `l_extendedprice` (line 149); `n = l_extendedprice.numel()` (line 87), and the four columns come from the same `lineitem`.
- `discount_ptr` (read, float64, length `n_elements`): `l_discount` (line 147).
- `quantity_ptr` (read, float64, length `n_elements`): `l_quantity` (line 148).
- `shipdate_ptr` (read, int32, length `n_elements`): `l_shipdate` in days (lines 144-146).
- `date_lo` (scalar, int32): from the date parameter `shipdate_lo` (line 141).
- `date_hi` (scalar, int32): from the date parameter `shipdate_hi` (line 142).
- `thresh_ptr` (read, float64, length `3`): `torch.tensor([discount_lo, discount_hi, quantity_hi])` (lines 89-93).
- `out_ptr` (readwrite, float64, length `1`): `torch.zeros(1)` (line 88); the kernel adds into it with `tl.atomic_add`.
- `n_elements` (size, int32): `l_extendedprice.numel()` (line 87), below 2^31 at scale factor 10, so `i32`.
- size symbol `n_elements`, least value 0.
