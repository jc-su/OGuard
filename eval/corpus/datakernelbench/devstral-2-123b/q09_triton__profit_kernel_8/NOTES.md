# q09_triton__profit_kernel_8

Kernel `_profit_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/full/queries/q09/round_02.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 2`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=8192`, `num_warps=8` (listed twice in the autotuner, with `num_stages=2` and `num_stages=4`).

## Extraction edits

- Renamed the kernel `_profit_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 8192}` and `NUM_WARPS = 8`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`_compute_amount` launches `_profit_kernel[grid]` at plan lines 61-64 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 60); `run_query` calls it at lines 109-114; the autotuner (lines 14-22) picks one of sixteen configurations, eight distinct in `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `l_extendedprice_ptr` (read, float64, length `n_elements`): `joined["l_extendedprice"]` (line 110); `n = l_extendedprice.numel()` (line 58), and the four columns come from the same `joined`.
- `l_discount_ptr` (read, float64, length `n_elements`): `joined["l_discount"]` (line 111).
- `ps_supplycost_ptr` (read, float64, length `n_elements`): `joined["ps_supplycost"]` (line 112).
- `l_quantity_ptr` (read, float64, length `n_elements`): `joined["l_quantity"]` (line 113).
- `amount_ptr` (write, float64, length `n_elements`): `torch.empty_like(l_extendedprice)` (line 59); the kernel only stores into it.
- `n_elements` (size, int32): `l_extendedprice.numel()` (line 58), below 2^31 at scale factor 10, so `i32`.
- size symbol `n_elements`, least value 0.
