# q22_triton__agg_kernel_7

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q22/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_agg_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_agg_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192, `num_warps` in 4, 8, and `num_stages` in 2, 4 (16 configurations, 8 distinct kernels). The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 7 of 8, `BLOCK_SIZE = 8192`, `num_warps = 4`.
- The two configurations of each `BLOCK_SIZE` and `num_warps` that differ only in `num_stages` give one corpus kernel: the corpus module states no `num_stages`, and the kernel has no loop for software pipelining, so both are this source.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_agg` launches `_agg_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = balance.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `bal_ptr`, `code_ptr`, `mask_ptr`: `c_acctbal`, the first two phone digits as int32, and the final mask (uint8) of `customer`, `n_elements = n` rows (`i32` at scale factor 10).
- `out_cnt`, `out_sum`: `torch.zeros(max_code + 1)` with `max_code = 99`, 100 entries each, reset to zero by the autotuner; the kernel adds into them atomically.
- `code_ptr`: the kept rows' codes are among the seven target codes (below 100) only because the plan's mask is derived from the codes, a relation between buffers; it is `read`.
