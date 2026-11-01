# q17_triton__final_sum_kernel_4

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q17/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_final_sum_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_final_sum_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 4 of 4, `BLOCK_SIZE = 8192`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_final_sum` launches `_final_sum_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = price.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `price_ptr`, `quantity_ptr`: `l_extendedprice` and `l_quantity` of `lineitem`, `n_elements = n` rows (`i32` at scale factor 10).
- `idx_ptr`: `torch.clamp(torch.searchsorted(sorted_keys, l_partkey), max=n_parts - 1)` over the same rows; `searchsorted` gives `[0, n_parts]` and the clamp `[0, n_parts - 1]` (with `n_parts = 0` the plan's `sorted_keys[idx_clamped]` fails for a non-empty lineitem), so every value is an index into `thresh_ptr`.
- `part_keep_ptr`: the membership flag `in_part` as uint8, `n` rows.
- `thresh_ptr`: `0.2 * avg_qty`, `n_parts` entries.
- `out_ptr`: `torch.zeros(1)`, reset to zero by the autotuner, added into once per program.
