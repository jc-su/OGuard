# q09_core_triton__amount_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q09/round_04.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_amount_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_amount_kernel` is renamed `kernel`.
- The plan passes `BLOCK_SIZE = 8192` positionally and no `num_warps`, so `CONSTANTS` states `BLOCK_SIZE = 8192` and `NUM_WARPS` the default 4.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_amount_kernel` launches `_amount_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_extendedprice.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[262144, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 262144 programs.

## Lengths and ranges

- `a_ptr`, `d_ptr`, `c_ptr`, `q_ptr`: `l_extendedprice`, `l_discount`, `ps_supplycost`, and `l_quantity` of the joined table, `n_elements = n` rows (`i32` at scale factor 10).
- `out_ptr`: `torch.empty_like(l_extendedprice)`, `n_elements` rows, stored at every row.
