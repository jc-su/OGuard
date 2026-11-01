# q14_cuda_q14_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q14/round_02.py`, copied as `plan.py`: `q14_kernel` in `_CUDA_SOURCE`, lines 15-62 of the plan file.

## Extraction edits

- `__global__ void q14_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses. The kernel uses `#define BLOCK_SIZE 1024`, copied unchanged.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q14_launcher` launches `q14_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = min((n + 1023) / 1024, 65536)`, when `n > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `price`, `discount`, `promo`: columns of the join of the filtered `lineitem` with `part`, `n = price.numel()` rows each, at least 1 at the launch.
- `out_promo_rev`, `out_total_rev`: `torch::zeros({1})`, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
