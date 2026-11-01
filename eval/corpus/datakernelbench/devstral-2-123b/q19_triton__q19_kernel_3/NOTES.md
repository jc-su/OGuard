# q19_triton__q19_kernel_3

Kernel `_q19_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/triton/full/queries/q19/round_02.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 2`), copied here as `plan.py`.

Configuration: `BLOCK_SIZE=4096`, `num_warps=4` (listed twice in the autotuner, with `num_stages=2` and `num_stages=4`).

## Extraction edits

- Renamed the kernel `_q19_kernel` to `kernel`.
- Replaced the `@triton.autotune` decorator by this configuration's values: `CONSTANTS = {"BLOCK_SIZE": 4096}` and `NUM_WARPS = 4`. The corpus module does not state `num_stages`; the kernel is loop-free, so pipelining leaves it unchanged, and the front end compiles it with Triton's default.
- Dropped the decorator's `reset_to_zero` list (`out_ptr`), which the autotuner zeroes between its trial runs; the plan allocates these buffers zeroed before the launch.
- Added `TYPES`, the Triton type of each argument's value at the plan's launch.
- Left out the host code.

## Launch

`run_query` launches `_q19_kernel[grid]` at plan lines 147-153 with `grid = (cdiv(n, BLOCK_SIZE),)` (line 146); the autotuner (lines 14-23) picks one of twelve configurations, six distinct in `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- `l_quantity_ptr` (read, float64, length `n_elements`): `joined["l_quantity"]` (line 136); `n = l_quantity_t.numel()` (line 143), and the six columns come from the same `joined`.
- `l_extendedprice_ptr` (read, float64, length `n_elements`): `joined["l_extendedprice"]` (line 137).
- `l_discount_ptr` (read, float64, length `n_elements`): `joined["l_discount"]` (line 138).
- `p_brand_id_ptr` (read, int32, length `n_elements`): the brand code of the same rows (lines 116-119, 139).
- `p_container_grp_ptr` (read, int32, length `n_elements`): the container group of the same rows (lines 122-134, 140).
- `p_size_ptr` (read, int32, length `n_elements`): `joined["p_size"]` (line 141).
- `brand12_id` (scalar, int32): the constant 12 (line 115).
- `brand23_id` (scalar, int32): the constant 23 (line 115).
- `brand34_id` (scalar, int32): the constant 34 (line 115).
- `sm_grp` (scalar, int32): the constant 1 (line 121).
- `md_grp` (scalar, int32): the constant 2 (line 121).
- `lg_grp` (scalar, int32): the constant 3 (line 121).
- `out_ptr` (readwrite, float64, length `1`): `torch.zeros(1)` (line 144); the kernel adds into it with `tl.atomic_add`.
- `n_elements` (size, int32): `l_quantity_t.numel()` (line 143), below 2^31 at scale factor 10, so `i32`.
- size symbol `n_elements`, least value 0.
