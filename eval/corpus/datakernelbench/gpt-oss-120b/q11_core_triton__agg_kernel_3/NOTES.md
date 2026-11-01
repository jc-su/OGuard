# q11_core_triton__agg_kernel_3

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q11/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_agg_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 2048`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_agg_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 and `num_warps` in 4, 8, all with `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 3 of 8, `BLOCK_SIZE = 2048`, `num_warps = 4`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_agg` launches `_agg_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = ps_supplycost.numel()`. This kernel has `BLOCK_SIZE = 2048`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[1048576, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 1048576 programs.

## Lengths and ranges

- `supply_ptr`, `avail_ptr`: `ps_supplycost` and `ps_availqty` (as float64) of the joined table, `n_elements = n` rows (`i32` at scale factor 10; the plan's comment says i64, but the value is a Python int below 2^31).
- `out_ptr`: `torch.empty_like(ps_supplycost)`, `n_elements` rows, stored at every row; `total_ptr`: `torch.zeros(1)`, reset to zero by the autotuner, which each program adds its block sum into atomically.
