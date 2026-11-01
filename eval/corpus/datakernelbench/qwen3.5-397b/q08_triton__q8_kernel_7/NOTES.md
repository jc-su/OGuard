# q08_triton__q8_kernel_7

The kernel `_q8_kernel` of the Triton plan for q08 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q08/round_01.py`,
copied as `plan.py`), in configuration 7 of 8: `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- `_q8_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `year_ptr: *i32`, `volume_ptr: *fp64`, `is_brazil_ptr: *i1`, `out_ptr: *fp64`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 120: `_K._q8_kernel[grid](...)` in `_launch_q8`, with `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The three columns have `n = o_year.numel()` rows; `is_brazil` is a bool tensor.
- `out` is `torch.zeros(4)`, an atomic accumulator: `readwrite`.
