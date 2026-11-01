# q03_triton__filter_pack_kernel_2

The kernel `_filter_pack_kernel` of the Triton plan for q03 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q03/round_03.py`,
copied as `plan.py`), in configuration 2 of 2: `BLOCK_SIZE = 4096`, `num_warps = 8`.

## Extraction edits

- `_filter_pack_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `lk_ptr: *i64`, `ep_ptr: *fp64`, `disc_ptr: *fp64`, `sd_ptr: *i32`, `od_ptr: *i32`, `sp_ptr: *i32`, `out_keys_ptr: *i64`, `out_rev_ptr: *fp64`, `block_offsets_ptr: *i32`, `sd_lo: i32`, `od_hi: i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 178: `_K._filter_pack_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE)` in `_run_triton_filter`, with `grid = (triton.cdiv(n, 4096),)`, `BLOCK_SIZE = 4096`, and the autotuner's `num_warps`.

## Lengths and ranges

- The columns are columns of the joined table `col`, of `n = lk.numel()` rows; the plan returns before the launches when `n == 0`.
- The plan passes `BLOCK_SIZE = 4096` to both launches; the autotuner's configurations vary `num_warps` and `num_stages` only.
- `out_keys` and `out_rev` are `torch.empty(total_count)`, with `total_count` the number of selected rows; the plan returns before this launch when it is 0.
- `start_offsets` is the exclusive prefix sum of `block_counts`, `grid_size = triton.cdiv(n, 4096)` elements, its own size symbol. The prefix-sum relation is not a range the plan states on its own, so it takes the role `read`.
