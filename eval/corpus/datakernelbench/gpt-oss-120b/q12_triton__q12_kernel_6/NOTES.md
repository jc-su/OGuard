# q12_triton__q12_kernel_6

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q12/round_05.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_q12_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 4096`, `num_warps = 8`.

## Extraction edits

- The entry kernel `_q12_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192, `num_warps` in 4, 8, and `num_stages` in 2, 4 (16 configurations, 8 distinct kernels). The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 6 of 8, `BLOCK_SIZE = 4096`, `num_warps = 8`.
- The two configurations of each `BLOCK_SIZE` and `num_warps` that differ only in `num_stages` give one corpus kernel: the corpus module states no `num_stages`, and the kernel has no loop for software pipelining, so both are this source.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_query_core` launches `_q12_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_shipmode_code.numel()`. This kernel has `BLOCK_SIZE = 4096`, `num_warps = 8`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `shipmode_ptr`, `priority_ptr`, `commit_ptr`, `receipt_ptr`, `ship_ptr`: columns of the joined table `joined`, `n_elements = n` rows (`i32` at scale factor 10).
- `out_high_ptr`, `out_low_ptr`: `torch.zeros(2, dtype=int64)` each, reset to zero by the autotuner, added into at indices 0 and 1.
