# q06_core_triton__q6_kernel_8

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q06/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_q6_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 8`.

## Extraction edits

- The entry kernel `_q6_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192, `num_warps` in 4, 8, and `num_stages` in 2, 4 (16 configurations, 8 distinct kernels). The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 8 of 8, `BLOCK_SIZE = 8192`, `num_warps = 8`.
- The two configurations of each `BLOCK_SIZE` and `num_warps` that differ only in `num_stages` give one corpus kernel: the corpus module states no `num_stages`, and the kernel has no loop for software pipelining, so both are this source.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_q6` launches `_q6_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_extendedprice.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 8`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[262144, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 262144 programs.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `quantity_ptr`, `shipdate_ptr`: columns of `lineitem` (`l_shipdate` as int32 days), `n_elements = n` rows (`i32` at scale factor 10).
- `thresh_ptr`: `torch.tensor([discount_lo, discount_hi, quantity_hi], float64)`, 3 entries, read at indices 0, 1, and 2.
- `out_ptr`: `torch.zeros(1)`, reset to zero by the autotuner; each program adds its block sum into it atomically.
