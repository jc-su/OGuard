# q19_triton__q19_kernel_7

The kernel `_q19_kernel` of the Triton plan for q19 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q19/round_02.py`,
copied as `plan.py`), in configuration 7 of 8: `brand12_id = 12`, `brand23_id = 23`, `brand34_id = 34`, `sm_grp = 1`, `md_grp = 2`, `lg_grp = 3`, `BLOCK_SIZE = 16384`, `num_warps = 8`.

## Extraction edits

- `_q19_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 1, 2; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `l_quantity_ptr: *fp64`, `l_extendedprice_ptr: *fp64`, `l_discount_ptr: *fp64`, `p_brand_id_ptr: *i32`, `p_container_grp_ptr: *i32`, `p_size_ptr: *i32`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 135: `_K._q19_kernel[grid](...)` in `_query_core`, when `n > 0`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)`, the `tl.constexpr` values `brand12_id = 12`, `brand23_id = 23`, `brand34_id = 34`, `sm_grp = 1`, `md_grp = 2`, `lg_grp = 3`, and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The six columns are columns of the joined table, of `n = l_quantity.numel()` rows, at least 1 at the launch.
- `out` is `torch.zeros(1)`, an atomic accumulator: `readwrite`.
