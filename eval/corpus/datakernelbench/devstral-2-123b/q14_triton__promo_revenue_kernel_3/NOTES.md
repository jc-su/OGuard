# q14_triton__promo_revenue_kernel_3

Kernel `_promo_revenue_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/full/queries/q14/round_02.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 2`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=16384`, `num_warps=8` (`num_stages=4`).

## Extraction edits

- Renamed the kernel `_promo_revenue_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 16384}` and `NUM_WARPS = 8`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Dropped the decorator's `reset_to_zero` list (`out_total` and `out_promo`), which the autotuner zeroes between its trial runs; the plan allocates these buffers zeroed before the launch.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`run_query` launches `_promo_revenue_kernel[grid]` at plan lines 107-110 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 106); the autotuner (lines 14-23) picks one of eight configurations.

## Lengths and ranges

- `extendedprice_ptr` (read, float64, length `n_elements`): `merged["l_extendedprice"]` (line 94); `n = extprice_t.numel()` (line 101), and the four columns come from the same `merged`.
- `discount_ptr` (read, float64, length `n_elements`): `merged["l_discount"]` (line 95).
- `shipdate_ptr` (read, int32, length `n_elements`): `l_shipdate` in days (lines 96-98).
- `promo_ptr` (read, bool, length `n_elements`): `merged["p_type_is_promo"]` as bool (line 99).
- `date_lo` (scalar, int32): from the date parameter `shipdate_lo` (line 84).
- `date_hi` (scalar, int32): from the date parameter `shipdate_hi` (line 85).
- `out_total` (readwrite, float64, length `1`): `torch.zeros(())` as float64 (line 102), one element; the kernel adds into it with `tl.atomic_add`.
- `out_promo` (readwrite, float64, length `1`): `torch.zeros(())` as float64 (line 103), as `out_total`.
- `n_elements` (size, int32): `extprice_t.numel()` (line 101), below 2^31 at scale factor 10, so `i32`.
- size symbol `n_elements`, least value 0.
