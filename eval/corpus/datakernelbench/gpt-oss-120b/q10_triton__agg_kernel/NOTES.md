# q10_triton__agg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q10/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_agg_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 1024`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_agg_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, then the same sizes with `num_warps=8`, all `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 1 of 8, `BLOCK_SIZE = 1024`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_query_core` launches `_agg_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = c_custkey.numel()`. This kernel has `BLOCK_SIZE = 1024`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `price_ptr`, `discount_ptr`: columns of the joined table `loc`, `n_elements = n` rows (`i32` at scale factor 10).
- `inv_ptr`: the inverse of `torch.unique(c_custkey, return_inverse=True)` over the same `n` rows, so every value lies in `[0, len(uniq_keys))`: an index into `out_ptr`.
- `out_ptr`: `torch.zeros(uniq_keys.shape[0])`, `n_keys` entries, reset to zero by the autotuner; the kernel adds into it atomically.
