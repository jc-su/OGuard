# q13_triton__count_orders_kernel_2

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q13/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_count_orders_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 2048`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_count_orders_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 2 of 4, `BLOCK_SIZE = 2048`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_count_orders` launches `_count_orders_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = o_custkey.numel()`. This kernel has `BLOCK_SIZE = 2048`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `o_custkey_ptr`, `mask_ptr`: `o_custkey` and the NOT LIKE mask (uint8) of `orders`, `n_elements = n` rows (`i32` at scale factor 10).
- `out_ptr`: `torch.zeros(num_customers + 1, dtype=int32)` with `num_customers = max(c_custkey)` of the customer table, reset to zero by the autotuner; the kernel adds into it atomically.
- `o_custkey_ptr`: the order keys are bounded by `max(c_custkey)` only through TPC-H's referential integrity, not by the plan, so it is `read`.
