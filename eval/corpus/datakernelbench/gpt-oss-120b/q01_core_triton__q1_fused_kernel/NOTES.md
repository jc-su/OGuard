# q01_core_triton__q1_fused_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q01/round_07.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_q1_fused_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 1024`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_q1_fused_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 1 of 4, `BLOCK_SIZE = 1024`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_q1` launches `_q1_fused_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_quantity.shape[0]`. This kernel has `BLOCK_SIZE = 1024`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[2097152, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 2097152 programs.

## Lengths and ranges

- `rf_ptr`, `ls_ptr`, `qty_ptr`, `price_ptr`, `disc_ptr`, `tax_ptr`, `shipdate_ptr`: columns of `lineitem` (`l_returnflag` and `l_linestatus` as cuDF category codes cast to int32, the numeric columns, `l_shipdate` as int32 days), `n_elements = n` rows (59,986,052 at scale factor 10, so `i32`).
- `out_sum_*`, `out_count`: `torch.zeros(MAX_GROUPS)` with `MAX_GROUPS = 256 * 256 = 65536` (float64, and int64 for the count), reset to zero by the autotuner; the kernel adds into them atomically.
- The plan does not bound the category codes, so `rf_ptr` and `ls_ptr` are `read`: the group index `rf * 256 + ls` stays below 65536 only while each column has at most 256 categories.
