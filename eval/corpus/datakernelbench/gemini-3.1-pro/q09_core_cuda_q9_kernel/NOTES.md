# q09_core_cuda_q9_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q09/round_01.py`, copied as `plan.py`: `q9_kernel` in `_CUDA_SOURCE`, lines 13-25 of the plan file.

## Extraction edits

- `__global__ void q9_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q9_launcher` launches `q9_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = (n + 1023) / 1024`, when `n > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_extendedprice`, `l_discount`, `ps_supplycost`, `l_quantity`: columns of the joined frame, `n_elements = l_extendedprice.numel()` rows each, at least 1 at the launch.
- `out`: `torch::empty({n})`, written by the kernel: write, `n_elements` elements.
