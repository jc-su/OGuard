# q04_core_triton__filter_lineitems_kernel_4

The kernel `_filter_lineitems_kernel` of the Triton plan for q04 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q04/round_01.py`,
copied as `plan.py`), in configuration 4 of 8: `BLOCK_SIZE = 2048`, `num_warps = 8`.

## Extraction edits

- `_filter_lineitems_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `line_key_ptr: *i64`, `line_commit_ptr: *i32`, `line_receipt_ptr: *i32`, `out_key_ptr: *i64`, `out_valid_ptr: *i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 150: `_K._filter_lineitems_kernel[grid](...)` in `_launch_filter_lineitems`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The three columns are lineitem columns of `n = l_orderkey.numel()` rows.
- `out_key` and `out_valid` are `torch.zeros(n)`; the kernel stores the kept rows' key and every row's flag: `write`.
