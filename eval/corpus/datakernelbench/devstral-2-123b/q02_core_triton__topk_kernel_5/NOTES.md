# q02_core_triton__topk_kernel_5

Kernel `_topk_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/core/queries/q02/round_06.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 6`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=4096`, `num_warps=4` (listed twice in the autotuner, with `num_stages=2` and `num_stages=4`).

## Extraction edits

- Renamed the kernel `_topk_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"limit": 100, "BLOCK_SIZE": 4096}` and `NUM_WARPS = 4`. The corpus module does not state `num_stages`; the kernel's `for i in range(min(limit, n_elements))` loop holds only scalar stores and no load, so pipelining leaves it unchanged (checked: the PTX for `num_stages=2` and `num_stages=4` is the same for all eight configurations), and the front end compiles it with Triton's default.
- Dropped the decorator's `reset_to_zero` list (`out_ptr`), which the autotuner zeroes between its trial runs; the plan allocates these buffers zeroed before the launch.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`_query_core` launches `_topk_kernel[grid]` at plan lines 74-78 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 73) and `limit` as the `tl.constexpr` argument; `run_query` calls it (lines 197-200) only when the final frame has rows, with its `limit` parameter, 100 by default (line 88), a query parameter that the plan fixes, so the `tl.constexpr` `limit` takes 100. The autotuner (lines 13-22) picks one of sixteen configurations, eight distinct in `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `s_acctbal_ptr` (read, float64, length `n_elements`): `s_acctbal_t` (line 198), the sorted final frame's `s_acctbal`; `n = s_acctbal.shape[0]` (line 66). Its loaded values are unused, so the PTX has no load.
- `p_partkey_ptr` (read, int64, length `n_elements`): `p_partkey_t` (line 199), the same frame; loaded and unused.
- `out_ptr` (write, int64, length `n_out`): `torch.zeros(size)` with `size = min(limit, n)` (lines 67, 70), `min(100, n_elements)`, which is not affine in the sizes; the signature gives it its own size symbol `n_out`, least value 1 (`n >= 1` at the launch).
- `out_size_ptr` (write, int64, length `1`): `torch.zeros(1)` (line 71).
- `n_elements` (size, int32): `n` (line 66), below 2^31 at scale factor 10, so `i32`; the plan launches only when `len(final) > 0` (line 197), so its least value is 1.
- `limit` (`tl.constexpr`): 100.
- size symbol `n_elements`, least value 1.
- size symbol `n_out`, least value 1.
