# q03_triton__revenue_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q03/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_revenue_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 1024`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_revenue_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, then the same sizes with `num_warps=8`, all `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 1 of 8, `BLOCK_SIZE = 1024`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_revenue` launches `_revenue_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_extendedprice.numel()`. This kernel has `BLOCK_SIZE = 1024`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `price_ptr`, `disc_ptr`: `l_extendedprice` and `l_discount` of the joined table `col`, `n_elements = n` rows (below 2^31 at scale factor 10, so `i32`).
- `out_rev_ptr`: `torch.empty_like(l_extendedprice)`, `n` rows, which the kernel only writes.
