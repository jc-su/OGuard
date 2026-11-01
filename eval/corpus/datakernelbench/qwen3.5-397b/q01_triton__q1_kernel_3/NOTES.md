# q01_triton__q1_kernel_3

The kernel `_q1_kernel` of the Triton plan for q01 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q01/round_07.py`,
copied as `plan.py`), in configuration 3 of 6: `BLOCK_SIZE = 4096`, `num_warps = 8`.

## Extraction edits

- `_q1_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 3, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `rf_ptr: *i32`, `ls_ptr: *i32`, `shipdate_ptr: *i32`, `qty_ptr: *fp64`, `price_ptr: *fp64`, `disc_ptr: *fp64`, `tax_ptr: *fp64`, `out_ptr: *fp64`, `shipdate_limit: i32`, `n_ls: i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 186: `_K._q1_kernel[grid](...)` in `run_query`, with `grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The seven columns are lineitem columns of `n_elements = l_quantity_t.numel()` rows (below 2^31 at scale factor 10: `i32`).
- `out_buffer` is `torch.zeros(n_groups * 6)` with `n_groups = n_rf * n_ls`, the product of the category counts of `l_returnflag` and `l_linestatus`: `6 * n_groups` elements, an atomic accumulator (`readwrite`). A launch has a non-empty lineitem, so `n_groups` is at least 1; the plan does not check that it is 6, the value of TPC-H's data.
