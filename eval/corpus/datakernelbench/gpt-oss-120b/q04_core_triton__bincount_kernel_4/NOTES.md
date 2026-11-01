# q04_core_triton__bincount_kernel_4

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q04/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_bincount_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_bincount_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 4 of 4, `BLOCK_SIZE = 8192`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_bincount` launches `_bincount_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = priority.numel()`; the plan returns before the launch when `n == 0`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[262144, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 262144 programs.

## Lengths and ranges

- `priority_ptr`: the `o_orderpriority` category codes (int32) of the orders that pass the date range and the EXISTS test, `n_elements = n` rows.
- `priority_ptr`: cuDF category codes lie in `[0, num_priorities)` with `num_priorities = len(priority_categories)`, the length of `out_ptr`: an index into `out_ptr`.
- `out_ptr`: `torch.zeros(num_priorities, dtype=int64)`, reset to zero by the autotuner; the kernel adds into it atomically, the masked lanes adding 0 at index 0.
