# q06_triton__q6_kernel

Kernel `_q6_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/full/queries/q06/round_01.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 1`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=1024`, `num_warps=4` (listed twice in the autotuner, with `num_stages=2` and `num_stages=4`).

## Extraction edits

- Renamed the kernel `_q6_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 1024}` and `NUM_WARPS = 4`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Dropped the decorator's `reset_to_zero` list (`out_ptr`), which the autotuner zeroes between its trial runs; the plan allocates these buffers zeroed before the launch.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`run_query` launches `_q6_kernel[grid]` at plan lines 103-106 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 102); the autotuner (lines 14-23) picks one of sixteen configurations, eight distinct in `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `price_ptr` (read, float64, length `n_elements`): `l_extendedprice` (line 94); `n = price_t.numel()` (line 99), and the four columns come from the same `lineitem`.
- `discount_ptr` (read, float64, length `n_elements`): `l_discount` (line 95).
- `quantity_ptr` (read, float64, length `n_elements`): `l_quantity` (line 96).
- `shipdate_ptr` (read, int32, length `n_elements`): `l_shipdate` in days (lines 88, 97).
- `date_lo` (scalar, int32): from the date parameter `shipdate_lo` (line 85).
- `date_hi` (scalar, int32): from the date parameter `shipdate_hi` (line 86).
- `thresh_ptr` (read, float64, length `3`): `torch.tensor([discount_lo, discount_hi, quantity_hi])` (line 101).
- `out_ptr` (readwrite, float64, length `1`): `torch.zeros(1)` (line 100); the kernel adds into it with `tl.atomic_add`.
- `n_elements` (size, int32): `price_t.numel()` (line 99), below 2^31 at scale factor 10, so `i32`.
- size symbol `n_elements`, least value 0.
