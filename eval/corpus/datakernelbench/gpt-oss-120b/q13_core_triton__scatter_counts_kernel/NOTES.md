# q13_core_triton__scatter_counts_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q13/round_04.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_scatter_counts_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_scatter_counts_kernel` is renamed `kernel`.
- The kernel declares `BLOCK_SIZE: tl.constexpr = 8192` and the plan passes no value and no `num_warps`, so `CONSTANTS` states `BLOCK_SIZE = 8192` and `NUM_WARPS` the default 4; the default in the parameter list is kept.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_scatter_counts` launches `_scatter_counts_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = o_custkey_filtered.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[262144, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 262144 programs.

## Lengths and ranges

- `custkey_ptr`: `o_custkey` of the orders whose comment passes the NOT LIKE filter, `n_elements = n` rows (`i32` at scale factor 10).
- `out_ptr`: `torch.zeros(num_customers + 1, dtype=int32)` with `num_customers = max(c_custkey)` of the customer table (1,500,000 at scale factor 10, so `max_key` is `i32`); the kernel adds into it atomically at the loaded key, an increment of 1 for keys in `(0, max_key]` and 0 otherwise.
- `custkey_ptr`: the order keys are bounded by `max(c_custkey)` only through TPC-H's referential integrity, not by the plan, so it is `read`.
