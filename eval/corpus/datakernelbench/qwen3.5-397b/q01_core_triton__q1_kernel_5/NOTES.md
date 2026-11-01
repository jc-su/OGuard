# q01_core_triton__q1_kernel_5

The kernel `_q1_kernel` of the Triton plan for q01 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/core/queries/q01/round_03.py`,
copied as `plan.py`), in configuration 5 of 8: `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_q1_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner,
  so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `rf_ptr: *i32`, `ls_ptr: *i32`, `qty_ptr: *fp64`, `price_ptr: *fp64`, `disc_ptr: *fp64`, `tax_ptr: *fp64`, `shipdate_ptr: *i32`, `shipdate_hi: i32`, `acc_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 116: `_K._q1_kernel[grid](...)` in `_query_core`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The seven columns are lineitem columns of `n = l_quantity.numel()` rows.
- `acc` is `torch.zeros(65536 * 6)`, 393216 doubles, an atomic accumulator: `readwrite`.
- `rf_ptr` and `ls_ptr` are cuDF category codes of `l_returnflag` and `l_linestatus` (three and two categories at TPC-H's data); the kernel's key `rf * 256 + ls` assumes both below 256, which no argument role states.
