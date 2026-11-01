# q02_core_triton__range_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q02/round_04.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_range_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 256`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_range_kernel` is renamed `kernel`.
- The plan launches the kernel with `BLOCK_SIZE=256` and no `num_warps`, so `CONSTANTS` states `BLOCK_SIZE = 256` and `NUM_WARPS` the default 4.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_range` launches `_range_kernel[grid]` with `grid = (cdiv(k, 256),)` and `n_elements = k = min(limit, len(final))`; the plan returns before the launch when `k == 0`. This kernel has `BLOCK_SIZE = 256`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[8388608, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 8388608 programs.

## Lengths and ranges

- `out_ptr`: `torch.empty(k, dtype=int64)`, `n_elements = k` entries (at most the `limit` parameter, 100 by default, so `i32`); the kernel stores `offs` at every row.
