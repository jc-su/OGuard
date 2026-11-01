# q19_core_triton__q19_kernel_2

The kernel `_q19_kernel` of the Triton plan for q19 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q19/round_02.py`,
copied as `plan.py`), in configuration 2 of 2: `BLOCK_SIZE = 8192`, `num_warps = 8`.

## Extraction edits

- `_q19_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `qty_ptr: *fp64`, `price_ptr: *fp64`, `disc_ptr: *fp64`, `brand_ptr: *i32`, `cont_ptr: *i32`, `size_ptr: *i32`, `brand12_id: i32`, `brand23_id: i32`, `brand34_id: i32`, `sm_grp: i32`, `md_grp: i32`, `lg_grp: i32`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 110: `_K._q19_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The six columns are columns of the joined lineitem and part table, of `n = l_quantity.numel()` rows.
- `out` is `torch.zeros(1)`, an atomic accumulator: `readwrite`.
