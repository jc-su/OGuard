# q19_cuda_q19_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q19/round_01.py`, copied as `plan.py`: `q19_kernel` in `_CUDA_SOURCE`, lines 13-56 of the plan file.

## Extraction edits

- `__global__ void q19_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses. The kernel uses `#define BLOCK_SIZE 1024`, copied unchanged.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q19_launcher` launches `q19_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = (n + 1023) / 1024`; it returns before the launch when `n == 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `price`, `discount`, `quantity`, `part_match`: columns of the join, `n = price.numel()` rows each, at least 1.
- `out`: `torch::zeros({1})`, zeroed and accumulated with `atomicAdd`: readwrite.
