# q06_cuda_q6_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q06/round_01.py`, copied as `plan.py`: `q6_kernel` in `_CUDA_SOURCE`, lines 15-66 of the plan file.

## Extraction edits

- `__global__ void q6_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses. The kernel uses `#define BLOCK_SIZE 1024`, copied unchanged.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q6_launcher` launches `q6_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = min((n + 1023) / 1024, 4096)`, when `n > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `price_ptr`, `discount_ptr`, `quantity_ptr`, `shipdate_ptr`: columns of `lineitem`, `n_elements = price.numel()` rows each, at least 1 at the launch.
- `out_ptr`: `torch::zeros({1})`, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
- The date bounds and the float bounds are any values.
