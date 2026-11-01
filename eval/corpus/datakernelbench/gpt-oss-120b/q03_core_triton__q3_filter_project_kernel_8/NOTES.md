# q03_core_triton__q3_filter_project_kernel_8

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q03/round_06.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_q3_filter_project_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 8`.

## Extraction edits

- The entry kernel `_q3_filter_project_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192, `num_warps` in 4, 8, and `num_stages` in 2, 4 (16 configurations, 8 distinct kernels). The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 8 of 8, `BLOCK_SIZE = 8192`, `num_warps = 8`.
- The two configurations of each `BLOCK_SIZE` and `num_warps` that differ only in `num_stages` give one corpus kernel: the corpus module states no `num_stages`, and the kernel has no loop for software pipelining, so both are this source.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_filter_project` launches `_q3_filter_project_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_orderkey.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 8`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[262144, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 262144 programs.

## Lengths and ranges

- `lk_ptr`, `price_ptr`, `discount_ptr`, `l_ship_ptr`, `o_date_ptr`, `sp_ptr`: columns of the joined table `col` (the dates as int32 days), `n_elements = n` rows (`i32` at scale factor 10; the plan's comment says int64, but the value is a Python int below 2^31).
- `out_lk_ptr`, `out_rev_ptr`, `out_od_ptr`, `out_sp_ptr`: `torch.empty(n)` each, the compacted kept rows, written at positions the counter hands out.
- `cnt_ptr`: `torch.zeros(1, dtype=int64)`, the compaction counter that each program adds its kept count to (`tl.atomic_add(cnt_ptr, block_sum)`) and reads the previous total from, reset to zero by the autotuner: `readwrite`. The write positions stay below `n` only because the counter starts at zero.
