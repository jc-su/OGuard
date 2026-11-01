# q12_triton__q12_kernel_2

The kernel `_q12_kernel` of the Triton plan for q12 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q12/round_03.py`,
copied as `plan.py`), in configuration 2 of 6: `BLOCK_SIZE = 1024`, `num_warps = 8`.

## Extraction edits

- `_q12_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 3, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `shipmode_ptr: *i32`, `priority_ptr: *i32`, `commit_ptr: *i32`, `receipt_ptr: *i32`, `ship_ptr: *i32`, `receipt_lo: i32`, `receipt_hi: i32`, `out_ptr: *i64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 108: `_K._q12_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The five columns are columns of the joined table, of `n = l_shipmode_code.numel()` rows.
- `out` is `torch.zeros(4)`, an atomic accumulator: `readwrite`.
