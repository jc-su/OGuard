# q21_core_triton__count_kernel_5

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q21/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_count_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 1024`, `num_warps = 8`.

## Extraction edits

- The entry kernel `_count_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, then the same with `num_warps=8`, all with `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 5 of 8, `BLOCK_SIZE = 1024`, `num_warps = 8`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_count_kernel` launches `_count_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l1_late.numel()`. This kernel has `BLOCK_SIZE = 1024`, `num_warps = 8`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[2097152, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 2097152 programs.

## Lengths and ranges

- `late_ptr`, `supp_ptr`, `distinct_ptr`, `late_distinct_ptr`: the `late` flag of `l1` (bool cast to int8), its `supp_id`, and the per-order aggregates gathered onto its rows, `n_elements = n` rows (`i32` at scale factor 10).
- `supp_ptr`: `supp_id = range(len(supp_n))` merged onto `l1`, so every value lies in `[0, num_suppliers)` with `num_suppliers = len(supp_n)`: an index into `out_ptr`.
- `out_ptr`: `torch.zeros(num_suppliers, dtype=int64)`, reset to zero by the autotuner; the kernel adds into it atomically under `keep`.
