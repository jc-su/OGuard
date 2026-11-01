# q05_core_triton__q5_agg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q05/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_q5_agg_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 1024`, `num_warps = 8`.

## Extraction edits

- The entry kernel `_q5_agg_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=8`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 1 of 4, `BLOCK_SIZE = 1024`, `num_warps = 8`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_agg` launches `_q5_agg_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = nation_id.numel()`. This kernel has `BLOCK_SIZE = 1024`, `num_warps = 8`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[2097152, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 2097152 programs.

## Lengths and ranges

- `nation_ptr`, `price_ptr`, `discount_ptr`: columns of the joined table `final_df`, `n_elements = n` rows (`i32` at scale factor 10). The plan annotates the pointer parameters with `tl.pointer_type(...)`; the annotations are kept.
- `nation_ptr`: `final_df` is merged with `nations_unique`, whose `nation_id` is `range(len(nations_unique))`, so every value lies in `[0, num_nations)`: an index into `out_ptr`.
- `out_ptr`: `torch.zeros(num_nations)` with `num_nations = len(nations_unique)`, reset to zero by the autotuner; the kernel adds into it atomically, the masked lanes adding 0.0 at index 0.
