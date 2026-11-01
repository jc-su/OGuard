# q15_core_triton__noop_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q15/round_10.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_noop_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 128`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_noop_kernel` is renamed `kernel`.
- The plan launches the kernel with `BLOCK_SIZE=128` and no `num_warps`, so `CONSTANTS` states `BLOCK_SIZE = 128` and `NUM_WARPS` the default 4.
- `TYPES` states the one argument that is not a `constexpr`, `n_elements`, as `i32`, its value being below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_noop` launches `_noop_kernel[grid]` with `grid = (cdiv(n, 128),)` and `n = l_extendedprice.numel()`; the query's work is done in PyTorch after the launch. This kernel has `BLOCK_SIZE = 128`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `n_elements`: the rows of `lineitem` (`i32` at scale factor 10), the kernel's only argument; it makes no memory access, so it needs no grid claim.
