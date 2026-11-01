# q04_triton__count_by_priority_kernel_8

Kernel `_count_by_priority_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/full/queries/q04/round_02.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 2`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=8192`, `num_warps=8` (listed twice in the autotuner, with `num_stages=2` and `num_stages=4`).

## Extraction edits

- Renamed the kernel `_count_by_priority_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 8192}` and `NUM_WARPS = 8`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Dropped the decorator's `reset_to_zero` list (`counts_ptr`), which the autotuner zeroes between its trial runs; the plan allocates these buffers zeroed before the launch.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`run_query` launches `_count_by_priority_kernel[grid]` at plan lines 110-112, only when `n_elements > 0` (line 108), with `grid = (cdiv(n_elements, BLOCK_SIZE),)` (line 109); the autotuner (lines 15-24) picks one of sixteen configurations, eight distinct in `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `priority_ptr` (index-in into `counts_ptr`, int32, length `n_elements`): `final_priority` (line 103), a selection of the `o_orderpriority` category codes (lines 70-71, 79), category codes index the `num_priorities` categories of the column (line 73), so each lies in `[0, num_priorities)`; `n_elements = final_priority.numel()` (line 106).
- `counts_ptr` (readwrite, int64, length `num_priorities`): `torch.zeros(num_priorities, dtype=torch.int64)` (line 107), `num_priorities` the number of categories (line 73); the kernel adds into it with `tl.atomic_add`.
- `n_elements` (size, int32): `final_priority.numel()` (line 106), at least 1 at the launch (line 108) and below 2^31, so `i32`.
- size symbol `n_elements`, least value 1: the plan launches only when `n_elements > 0` (line 108).
- size symbol `num_priorities`, least value 1: a launch has at least one row, whose code is one of the categories.
