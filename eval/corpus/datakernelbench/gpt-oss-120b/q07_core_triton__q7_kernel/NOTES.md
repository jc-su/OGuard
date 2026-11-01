# q07_core_triton__q7_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/core/queries/q07/round_06.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_q7_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_q7_kernel` is renamed `kernel`.
- The kernel declares `BLOCK_SIZE: tl.constexpr = 8192` and the plan passes no value and no `num_warps`, so `CONSTANTS` states `BLOCK_SIZE = 8192` and `NUM_WARPS` the default 4; the default in the parameter list is kept.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_q7` launches `_q7_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = supp_nation_code.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 4`; the block is `32 * num_warps` threads. `triton.json` states the grid claim `[262144, 1, 1]`: the kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in 32 bits, which does not wrap for at most 2^31 / BLOCK_SIZE = 262144 programs.

## Lengths and ranges

- `supp_ptr`, `cust_ptr`, `shipdate_ptr`, `price_ptr`, `disc_ptr`: columns of the joined table `joined` (`l_shipdate` as int32 days), `n_elements = n` rows (`i32` at scale factor 10; the plan's comment says int64, but the value is a Python int below 2^31).
- `supp_ptr`, `cust_ptr`: `(nation == nation_b).astype(int32)`, 0 or 1; the plan establishes this range only through the comparison, and no buffer has 2 entries, so both are `read`.
- `out_ptr`: `torch.zeros(8)`, the 8 groups `supp * 4 + cust * 2 + year_offset`; the kernel adds into it atomically.
- `date_lo`, `date_hi`, `year_boundary`: days since epoch, any int32.
