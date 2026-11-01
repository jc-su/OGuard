# q17_triton__accumulate_kernel_8

The kernel `_accumulate_kernel` of the Triton plan for q17 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q17/round_01.py`,
copied as `plan.py`), in configuration 8 of 8: `BLOCK_SIZE = 8192`, `num_warps = 8`.

## Extraction edits

- `_accumulate_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `partkey_ptr: *i64`, `quantity_ptr: *fp64`, `is_filtered_ptr: *i32`, `sum_qty_ptr: *fp64`, `cnt_qty_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 161: `_K._accumulate_kernel[grid](...)` in `run_query`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The lineitem columns have `n = l_partkey.numel()` rows; the plan returns before the launches when lineitem is empty.
- `is_filtered`, `sum_qty`, `cnt_qty`, and `thresholds` are `torch.zeros(array_size)` with `array_size = max(l_partkey) + 1`: `max_partkey + 1` elements. The plan sizes them from the largest `l_partkey`, so `partkey_ptr` is `index-in` `is_filtered_ptr` (partkeys are TPC-H's non-negative keys).
- `sum_qty` and `cnt_qty` are atomic accumulators: `readwrite`.
