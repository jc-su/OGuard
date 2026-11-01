# q04_core_triton__filter_orders_kernel_5

The kernel `_filter_orders_kernel` of the Triton plan for q04 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q04/round_01.py`,
copied as `plan.py`), in configuration 5 of 8: `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_filter_orders_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `order_key_ptr: *i64`, `order_date_ptr: *i32`, `priority_ptr: *i32`, `out_key_ptr: *i64`, `out_priority_ptr: *i32`, `out_valid_ptr: *i32`, `date_lo: i32`, `date_hi: i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 129: `_K._filter_orders_kernel[grid](...)` in `_launch_filter_orders`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The three columns are orders columns of `n = o_orderkey.numel()` rows.
- `out_key`, `out_priority`, and `out_valid` are `torch.zeros(n)`; the kernel stores the kept rows' key and priority and every row's flag: `write`.
