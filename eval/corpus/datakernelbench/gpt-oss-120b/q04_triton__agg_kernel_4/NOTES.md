# q04_triton__agg_kernel_4

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q04/round_05.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_agg_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_agg_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 4 of 4, `BLOCK_SIZE = 8192`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_agg` launches `_agg_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = o_orderkey.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `o_orderkey_ptr`, `o_orderdate_ptr`, `o_priority_ptr`: columns of `orders`, `n_elements = n` rows (`i32` at scale factor 10).
- `o_priority_ptr`: the cuDF category codes of `o_orderpriority` (converted with `to_cupy`, which refuses nulls), so every value lies in `[0, num_priorities)` with `num_priorities = len(categories)`: an index into `out_counts_ptr`.
- `bitmap_ptr`: the bitmap of the first kernel, `max_key + 1` entries (`bitmap_size`), read here.
- `out_counts_ptr`: `torch.zeros(num_priorities, dtype=int64)`, reset to zero by the autotuner; the kernel adds into it atomically.
- `o_orderkey_ptr`: bounded above by the length the plan gives the bitmap, with no lower bound, so it is `read`.
