# q15_triton__q15_kernel_4

Kernel `_q15_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/full/queries/q15/round_08.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 8`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=2048`, `num_warps=8` (`num_stages=1`).

## Extraction edits

- Renamed the kernel `_q15_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 2048}` and `NUM_WARPS = 8`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Dropped the decorator's `reset_to_zero` list (`totals_ptr` and `present_ptr`), which the autotuner zeroes between its trial runs; the plan allocates these buffers zeroed before the launch.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`run_query` launches `_q15_kernel[grid]` at plan lines 121-125 with `grid = (cdiv(n_line, BLOCK_SIZE),)` (line 120); the autotuner (lines 15-23) picks one of eight configurations.

## Lengths and ranges

- `l_suppkey_ptr` (read, int64, length `n_elements`): `lineitem["l_suppkey"]` (line 101); `n_line = l_price_t.numel()` (line 107), and the four columns come from the same `lineitem`.
- `l_price_ptr` (read, float64, length `n_elements`): `lineitem["l_extendedprice"]` (line 102).
- `l_disc_ptr` (read, float64, length `n_elements`): `lineitem["l_discount"]` (line 103).
- `l_shipdate_ptr` (read, int32, length `n_elements`): `l_shipdate` in days as int32 (line 104).
- `suppkey_map_ptr` (read, int32, length `max_suppkey`): `torch.full((max_suppkey,), -1)` with `max_suppkey = max(s_suppkey) + 1` (lines 112-113), then each `s_suppkey` set to its row in `supplier` (line 114), so its values lie in `[-1, n_supp)`; the -1 entries lie outside the index role's range `[0, n_supp)`, so the role is `read`. The kernel reads it at `l_suppkey`, which the plan does not bound.
- `shipdate_lo` (scalar, int32): from the date parameter `shipdate_lo` (line 98).
- `shipdate_hi` (scalar, int32): from the date parameter `shipdate_hi` (line 99).
- `totals_ptr` (readwrite, float64, length `n_supp`): `torch.zeros(n_supp)` with `n_supp = s_suppkey_t.numel()` (lines 108, 116); the kernel adds into it with `tl.atomic_add`.
- `present_ptr` (readwrite, int32, length `n_supp`): `torch.zeros(n_supp, dtype=torch.int32)` (line 117), as `totals_ptr`.
- `n_elements` (size, int32): `n_line = l_price_t.numel()` (line 107), below 2^31 at scale factor 10, so `i32`.
- size symbol `n_elements`, least value 0.
- size symbol `max_suppkey`, least value 1: `s_suppkey_t.max()` (line 112) needs a supplier row, and `suppkey_map[s_suppkey_t]` (line 114) needs that key inside the map.
- size symbol `n_supp`, least value 1: `s_suppkey_t.max()` (line 112) needs a supplier row.
