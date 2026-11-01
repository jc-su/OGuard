# q01_triton__q1_fused_kernel

Kernel `_q1_fused_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/full/queries/q01/round_08.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 8`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=2048`, `num_warps=8` (`num_stages=2`).

## Extraction edits

- Renamed the kernel `_q1_fused_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 2048}` and `NUM_WARPS = 8`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`run_query` launches `_q1_fused_kernel[grid]` at plan lines 116-121 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 104); the autotuner (lines 14-21) picks one of three configurations.

## Lengths and ranges

- `rf_ptr` (read, int32, length `n_elements`): the `l_returnflag` category codes as int32 (line 95); `n = qty.numel()` (line 103), and every column comes from the same `lineitem`.
- `ls_ptr` (read, int32, length `n_elements`): the `l_linestatus` category codes as int32 (line 96).
- `qty_ptr` (read, float64, length `n_elements`): `l_quantity` (line 97).
- `price_ptr` (read, float64, length `n_elements`): `l_extendedprice` (line 98).
- `disc_ptr` (read, float64, length `n_elements`): `l_discount` (line 99).
- `tax_ptr` (read, float64, length `n_elements`): `l_tax` (line 100).
- `ship_ptr` (read, int32, length `n_elements`): `l_shipdate` in days as int32 (line 101).
- `ship_hi` (scalar, int32): from the date parameter `shipdate_hi` (line 86); an int below 2^31, so `i32`.
- `n_elements` (size, int32): `qty.numel()` (line 103), below 2^31 at scale factor 10, so `i32`.
- `sum_qty_ptr` (write, float64, length `n_elements`): `torch.zeros(n)` (line 107); the kernel only stores into it.
- `sum_base_price_ptr` (write, float64, length `n_elements`): `torch.zeros(n)` (line 108).
- `sum_disc_price_ptr` (write, float64, length `n_elements`): `torch.zeros(n)` (line 109).
- `sum_charge_ptr` (write, float64, length `n_elements`): `torch.zeros(n)` (line 110).
- `avg_qty_ptr` (write, float64, length `n_elements`): `torch.zeros(n)` (line 111).
- `avg_price_ptr` (write, float64, length `n_elements`): `torch.zeros(n)` (line 112).
- `avg_disc_ptr` (write, float64, length `n_elements`): `torch.zeros(n)` (line 113).
- `count_ptr` (write, float64, length `n_elements`): `torch.zeros(n)` (line 114).
- size symbol `n_elements`, least value 0.
