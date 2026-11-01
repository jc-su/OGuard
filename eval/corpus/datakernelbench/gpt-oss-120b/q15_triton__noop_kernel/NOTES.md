# q15_triton__noop_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/triton/full/queries/q15/round_06.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the Triton kernel `_noop_kernel` of the plan's `_KERNEL_SOURCE`, with `N = 1`, `num_warps = 4`.

## Extraction edits

- The entry kernel `_noop_kernel` is renamed `kernel`.
- The plan launches the kernel with `N = 1`, `num_warps = 4`, the `constexpr` in `CONSTANTS`.
- `TYPES` states each argument that is not a `constexpr` with the Triton type of the plan's value: the tensor's element type for a pointer, and `i32` for each integer, all below 2^31 at scale factor 10.
- Copied unchanged: `import triton`, `import triton.language as tl`, and the `@triton.jit` function. The loader, the launch helpers, and the query code are left out as host code.

## Launch

`_launch_noop` launches `_noop_kernel[(1,)](dummy, 1)`: one program, the default `num_warps=4`, and the `constexpr` `N = 1`. This kernel has `N = 1`, `num_warps = 4`; the block is `32 * num_warps` threads.

## Lengths and ranges

- `dummy_ptr`: `torch.zeros(1, dtype=float64)`, which the kernel only writes.
