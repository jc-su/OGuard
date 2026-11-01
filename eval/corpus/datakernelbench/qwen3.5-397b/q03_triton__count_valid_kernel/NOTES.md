# q03_triton__count_valid_kernel

The kernel `_count_valid_kernel` of the Triton plan for q03 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/triton/full/queries/q03/round_03.py`,
copied as `plan.py`), in configuration 1 of 2: `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- `_count_valid_kernel` is renamed `kernel`.
- The `@triton.autotune` decorator is replaced by this configuration's values,
  `CONSTANTS` and `NUM_WARPS`. The autotuner lists this configuration with
  `num_stages` 2, 4; the front end does not take `num_stages`, and the kernel
  has no loop for the software pipeliner, so these give one corpus kernel.
- `TYPES` gives each argument that is not a `tl.constexpr` the Triton type of the
  plan's value: `sd_ptr: *i32`, `od_ptr: *i32`, `block_counts_ptr: *i32`, `sd_lo: i32`, `od_hi: i32`, `n_elements: i32`.
- No access, mask, index, or type is changed.

## Launch

`plan.py` line 148: `_K._count_valid_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE)` in `_run_triton_filter`, with `grid = (triton.cdiv(n, 4096),)`, `BLOCK_SIZE = 4096`, and the autotuner's `num_warps`.

## Lengths and ranges

- The columns are columns of the joined table `col`, of `n = lk.numel()` rows; the plan returns before the launches when `n == 0`.
- The plan passes `BLOCK_SIZE = 4096` to both launches; the autotuner's configurations vary `num_warps` and `num_stages` only.
- `block_counts` is `torch.zeros(grid_size)` with `grid_size = triton.cdiv(n, 4096)`, one count per program of the plan's grid; the length is a quotient, so it is its own size symbol. Each program stores its count: `write`.
