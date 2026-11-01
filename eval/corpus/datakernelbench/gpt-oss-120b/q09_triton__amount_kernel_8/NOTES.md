# q09_triton__amount_kernel_8

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q09/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_amount_kernel` of the plan's `_KERNEL_SOURCE`, with `BLOCK_SIZE = 8192`, `num_warps = 8`.

## Extraction edits

- The entry kernel `_amount_kernel` is renamed `kernel`.
- The plan autotunes the kernel over `BLOCK_SIZE` in 1024, 2048, 4096, 8192 with `num_warps=4`, then the same sizes with `num_warps=8`, all `num_stages=2`. The `@triton.autotune` decorator is replaced by this configuration's values, `CONSTANTS` and `NUM_WARPS`: configuration 8 of 8, `BLOCK_SIZE = 8192`, `num_warps = 8`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_amount` launches `_amount_kernel[grid]` with `grid = (cdiv(n, BLOCK_SIZE),)` and `n = l_extendedprice.numel()`. This kernel has `BLOCK_SIZE = 8192`, `num_warps = 8`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `ext_ptr`, `disc_ptr`, `sup_ptr`, `qty_ptr`: columns of the joined table `joined`, `n_elements = n` rows (`i32` at scale factor 10).
- `out_ptr`: `torch.empty(n)`, which the kernel only writes.
