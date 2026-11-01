# q18_core_triton__agg_kernel_2

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q18/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_agg_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 2048`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_agg_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 2 of 4, `BLOCK_SIZE = 2048`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_query_core` launches `_agg_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_orderkey.numel()`. This kernel has `BLOCK_SIZE = 2048`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[1048576, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 1048576 programs.

## Lengths and ranges

- `orderkey_ptr`, `qty_ptr`: `l_orderkey` and `l_quantity` of `lineitem`, `n_elements = n` rows (`i32` at scale factor 10; the plan's comment says int64, but the value is a Python int below 2^31).
- `sums_ptr`: `torch.zeros(max_key + 1)` with `max_key = max(l_orderkey)`, reset to zero by the autotuner; the kernel adds into it atomically at the loaded order key. Its length is the size symbol `max_key + 1`.
- `orderkey_ptr`: the plan bounds its values only above, by the length it gives `sums_ptr`, and establishes no lower bound (TPC-H order keys are positive), so it is `read`, not an index.
