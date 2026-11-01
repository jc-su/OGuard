# q12_core_triton__q12_kernel_6

Kernel `_q12_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/core/queries/q12/round_05.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 5`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=4096`, `num_warps=8` (listed twice in the autotuner, with `num_stages=2` and `num_stages=4`).

## Extraction edits

- Renamed the kernel `_q12_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 4096}` and `NUM_WARPS = 8`. The corpus module does not state `num_stages`; the kernel's only loop is a `tl.static_range(2)`, unrolled at compile time, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Dropped the decorator's `reset_to_zero` list (`high_out_ptr`, `low_out_ptr`), which the autotuner zeroes between its trial runs; the plan allocates these buffers zeroed before the launch.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`_run_q12_kernel` launches `_q12_kernel[grid]` at plan lines 88-92 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 87); `_query_core` calls it at lines 106-109 and `run_query` at lines 164-172. The autotuner (lines 14-23) picks one of sixteen configurations, eight distinct in `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `l_commitdate_ptr` (read, int32, length `n_elements`): `l_commitdate` in days (lines 152-154); `n = l_commitdate.numel()` (line 83), and the five columns come from the same `joined` (line 143).
- `l_receiptdate_ptr` (read, int32, length `n_elements`): lines 155-157.
- `l_shipdate_ptr` (read, int32, length `n_elements`): lines 158-160.
- `shipmode_ptr` (read, int32, length `n_elements`): `shipmode_code` (lines 146, 161), 0 or 1; the kernel compares it with the static loop variable and does not index by it.
- `o_orderpriority_ptr` (read, int32, length `n_elements`): `priority_high` (lines 147-150, 162), 0 or 1.
- `receipt_lo`, `receipt_hi` (scalar, int32): the day counts of the parameters `receipt_lo` and `receipt_hi` (lines 137-138).
- `high_out_ptr` (readwrite, int64, length `2`): `torch.zeros(2)` (line 84); the kernel adds into it with `tl.atomic_add`.
- `low_out_ptr` (readwrite, int64, length `2`): `torch.zeros(2)` (line 85), as `high_out_ptr`.
- `n_elements` (size, int32): `l_commitdate.numel()` (line 83), below 2^31 at scale factor 10, so `i32`.
- size symbol `n_elements`, least value 0.
