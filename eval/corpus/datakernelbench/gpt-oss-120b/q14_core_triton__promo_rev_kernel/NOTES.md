# q14_core_triton__promo_rev_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q14/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_promo_rev_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 2048`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_promo_rev_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 2048, 4096, 8192, 16384 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 1 of 4, `BLOCK_SIZE = 2048`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_promo_rev` launches `_promo_rev_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_extendedprice.numel()`. This kernel has `BLOCK_SIZE = 2048`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[1048576, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 1048576 programs.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `shipdate_ptr`, `promo_ptr`: columns of the joined table (`l_shipdate` as int32 days, the PROMO flag as bool cast to uint8), `n_elements = n` rows (`i32` at scale factor 10).
- `out_total_ptr`, `out_promo_ptr`: `torch.zeros(1)` each, reset to zero by the autotuner; each program adds its block sums into them atomically.
