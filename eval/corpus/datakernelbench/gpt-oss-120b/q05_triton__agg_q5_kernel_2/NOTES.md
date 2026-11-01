# q05_triton__agg_q5_kernel_2

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q05/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_agg_q5_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 2048`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_agg_q5_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 2 of 4, `BLOCK_SIZE = 2048`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_agg_q5` launches `_agg_q5_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = nation_id.numel()`. This kernel has `BLOCK_SIZE = 2048`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `nation_ptr`, `price_ptr`, `discount_ptr`: columns of the joined table `final_df`, `n_elements = n` rows (`i32` at scale factor 10).
- `nation_ptr`: `final_df` is merged with `nations_unique`, whose `nation_id` is `range(len(nations_unique))`, so every value lies in `[0, num_nations)`: an index into `out_ptr`.
- `out_ptr`: `torch.zeros(num_nations)`, reset to zero by the autotuner; the kernel adds into it atomically.
