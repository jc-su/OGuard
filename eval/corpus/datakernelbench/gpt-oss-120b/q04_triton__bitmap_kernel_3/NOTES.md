# q04_triton__bitmap_kernel_3

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q04/round_05.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_bitmap_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 4096`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_bitmap_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 3 of 4, `BLOCK_SIZE = 4096`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_bitmap` launches `_bitmap_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_orderkey.numel()`. This kernel has `BLOCK_SIZE = 4096`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `l_orderkey_ptr`, `l_commit_ptr`, `l_receipt_ptr`: `l_orderkey` (int64) and the two dates as int32 days of `lineitem`, `n_elements = n` rows (`i32` at scale factor 10).
- `bitmap_ptr`: `torch.zeros(max_key + 1, dtype=int32)` with `max_key` the largest `o_orderkey` or `l_orderkey`, reset to zero by the autotuner; the kernel updates it with `atomic_max`. Its length is the size symbol `bitmap_size`.
- `l_orderkey_ptr`: bounded above by the length the plan gives the bitmap, with no lower bound, so it is `read`, not an index.
