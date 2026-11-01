# q19_core_cuda_q19_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q19/round_01.py`, copied as `plan.py`: `q19_kernel` in `_CUDA_SOURCE`, lines 12-71 of the plan file.

## Extraction edits

- `__global__ void q19_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q19_launcher` launches `q19_kernel<<<blocks, threads>>>` with `threads = BLOCK_SIZE = 1024` and `blocks = (numel + 1023) / 1024`, when `numel > 0`. The kernel uses `#define BLOCK_SIZE 1024`, copied unchanged.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `quantity_ptr`, `extendedprice_ptr`, `discount_ptr`, `brand_id_ptr`, `container_grp_ptr`, `size_ptr`: columns of the join of the filtered `lineitem` with the filtered `part`, `numel = quantity.numel()` rows each, at least 1 at the launch.
- `out_ptr`: `torch::zeros({1})`, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
- The brand and container codes are compared only.
