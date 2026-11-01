# q19_core_triton__revenue_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q19/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_revenue_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 1024`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_revenue_kernel` is renamed `kernel`.
- The plan passes `BLOCK_SIZE = 1024` positionally and no `num_warps`, so `CONSTANTS` states `BLOCK_SIZE = 1024` and `NUM_WARPS` the default 4.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_revenue` launches `_revenue_kernel[grid]` with `grid = (num_blocks,)`, `num_blocks = (n + 1023) // 1024`, and `n = l_quantity.shape[0]`. This kernel has `BLOCK_SIZE = 1024`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[2097152, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 2097152 programs.

## Lengths and ranges

- `l_quantity_ptr`, `l_extendedprice_ptr`, `l_discount_ptr`, `p_brand_id_ptr`, `p_container_grp_ptr`, `p_size_ptr`: columns of the joined table (the brand and container as int32 codes), `n_elements = n` rows (`i32` at scale factor 10).
- `brand12_id`, `brand23_id`, `brand34_id`, `sm_grp`, `md_grp`, `lg_grp`: the codes the plan compares against, any int32.
- `out_ptr`: `torch.zeros(num_blocks)` with `num_blocks = (n + 1023) // 1024`, one entry per program of the plan's launch, which each program stores its block sum into at `pid`. The length is a quotient, not affine in `n_elements`, so it is the size symbol `out_ptr`; the store stays inside it only for a grid of at most `num_blocks` programs, a bound on the grid that depends on a size and that the claim cannot state.
