# q11_triton__q11_accumulate_kernel_6

The kernel `_q11_accumulate_kernel` of the Triton plan for q11 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q11/round_01.py`,
copied as `plan.py`), in configuration 6 of 8: `BLOCK_SIZE = 4096`, `num_warps = 8`.

## Extraction edits

- `_q11_accumulate_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `supplycost_ptr: *fp64`, `availqty_ptr: *fp64`, `inverse_ptr: *i64`, `out_ptr: *fp64`, `n_elements: i32`, `n_groups: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 94: `_K._q11_accumulate_kernel[grid](...)` in `_launch_q11_accumulate`, with `grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)` and the autotuner's `BLOCK_SIZE` and `num_warps`.

## Lengths and ranges

- The columns have `n_elements = ps_supplycost.numel()` rows; `inverse` comes from `torch.unique(ps_partkey, return_inverse=True)` over as many rows.
- `out` is `torch.zeros(n_groups)` with `n_groups = unique_keys.shape[0]`, passed as the kernel's `n_groups`; an atomic accumulator (`readwrite`). The inverse indices lie in `[0, n_groups)`: `inverse_ptr` is `index-in` `out_ptr`.
