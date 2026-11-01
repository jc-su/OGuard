# q17_core_triton__filter_sum_kernel_3

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q17/round_06.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_filter_sum_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_filter_sum_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 3 of 4, `BLOCK_SIZE = 4096`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_q17_fused` launches `_filter_sum_kernel[grid]` with the same `grid = (cdiv(n_rows, BLOCK_SIZE),)` and `n_rows = sel_idx.numel()`, after `_agg_kernel`. This kernel has `BLOCK_SIZE = 4096`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[524288, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 524288 programs.

## Lengths and ranges

- `idx_ptr`, `qty_ptr`, `price_ptr`: the clamped `searchsorted` positions of the lineitem rows whose part is in the filtered set, and their `l_quantity` and `l_extendedprice`, `n_rows` rows (`i32` at scale factor 10; the plan's comment says int64, but the value is a Python int below 2^31); the plan returns before the launches when there is no such row.
- `idx_ptr`: `torch.clamp(idx, max=n_parts - 1)[in_part]`, so every value lies in `[0, n_parts)` with `n_parts = sorted_keys.numel()`: an index into `thresh_ptr`.
- `thresh_ptr`: `0.2 * avg_qty`, `n_parts` entries; `out_ptr`: `torch.zeros(1)`, reset to zero by the autotuner, which each program adds its block sum into atomically.
