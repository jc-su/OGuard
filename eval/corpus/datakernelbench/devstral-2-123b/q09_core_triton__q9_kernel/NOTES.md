# q09_core_triton__q9_kernel

Kernel `_q9_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/core/queries/q09/round_03.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 3`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=1024`, `num_warps=4` (listed twice in the autotuner, with `num_stages=2` and `num_stages=4`).

## Extraction edits

- Renamed the kernel `_q9_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 1024}` and `NUM_WARPS = 4`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`_launch_q9` launches `_q9_kernel[grid]` at plan lines 60-63 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 59); `_query_core` calls it at line 72 and `run_query` at lines 116-121. The autotuner (lines 13-21) picks one of sixteen configurations, eight distinct in `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `extendedprice_ptr` (read, float64, length `n_elements`): `joined["l_extendedprice"]` (line 117); `n = l_extendedprice.numel()` (line 57), and the four columns come from the same `joined` (lines 110-114).
- `discount_ptr` (read, float64, length `n_elements`): line 118.
- `supplycost_ptr` (read, float64, length `n_elements`): line 119.
- `quantity_ptr` (read, float64, length `n_elements`): line 120.
- `out_ptr` (write, float64, length `n_elements`): `torch.empty_like(l_extendedprice)` (line 58).
- `n_elements` (size, int32): `l_extendedprice.numel()` (line 57), below 2^31 at scale factor 10, so `i32`.
- size symbol `n_elements`, least value 0.
