# q21_triton__q21_kernel_8

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q21/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_q21_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 8`.

## Extraction edits

- The entry kernel `_q21_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, then the same sizes with `num_warps=8`, all `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 8 of 8, `BLOCK_SIZE = 8192`, `num_warps = 8`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_q21` launches `_q21_kernel[grid]` with `grid = (cdiv(n_rows, BLOCK_SIZE),)` and `n_rows = supp_id.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 8`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `supp_id_ptr`, `late_ptr`, `distinct_ptr`, `late_dist_ptr`: columns of `l1` (the late flag as int8, the per-order aggregates merged in), `n_rows` rows (`i32` at scale factor 10).
- `supp_id_ptr`: `supp_n["supp_id"] = range(len(supp_n))`, merged into `l1` by an inner join, so every value lies in `[0, num_suppliers)`: an index into `out_ptr`.
- `out_ptr`: `torch.zeros(num_suppliers, dtype=int64)` with `num_suppliers = len(supp_n)`, reset to zero by the autotuner; the kernel adds into it atomically under `keep`.
