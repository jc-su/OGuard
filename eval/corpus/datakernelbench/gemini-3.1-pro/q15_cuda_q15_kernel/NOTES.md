# q15_cuda_q15_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q15/round_02.py`, copied as `plan.py`: `q15_kernel` in `_CUDA_SOURCE`, lines 12-36 of the plan file.

## Extraction edits

- `__global__ void q15_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses. The kernel uses `#include <cuda_runtime.h>`, copied unchanged.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q15_launcher` launches `q15_kernel<<<blocks, 256>>>` with `blocks = (n + 255) / 256`; the plan does not check `n > 0`.

`launch.json` states the block (256, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_suppkey`, `l_price`, `l_discount`, `l_shipdate`: columns of `lineitem`, `n_elements = l_suppkey.numel()` rows each.
- `out_revenue`, `out_present`: `torch.zeros(max_supp + 1)` with `max_supp = int(s_suppkey.max())` of `supplier`, zeroed by the plan; `out_revenue` is accumulated with `atomicAdd` and `out_present` keeps 0 at keys without a matching row: readwrite.
- `l_suppkey` refers to `supplier` by a foreign key; the plan sizes the outputs by `supplier`, not by this column: `read`.
