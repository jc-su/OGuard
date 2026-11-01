# q02_core_cuda_q2_arange_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q02/round_01.py`, copied as `plan.py`: `q2_arange_kernel` in `_CUDA_SOURCE`, lines 11-16 of the plan file.

## Extraction edits

- `__global__ void q2_arange_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q2_launcher` launches `q2_arange_kernel<<<blocks, threads>>>` with `threads = 256` and `blocks = (k + 255) / 256`, when `k > 0`.

`launch.json` states the block (256, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `out`: `torch::empty({k})` with `k = min(limit, n)`, the rows to return, written by the kernel: write; `k` is its length (size), at least 1 at the launch.
