# q12_cuda_q12_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q12/round_02.py`, copied as `plan.py`: `q12_kernel` in `_CUDA_SOURCE`, lines 13-80 of the plan file.

## Extraction edits

- `__global__ void q12_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses. The kernel uses `#define BLOCK_SIZE 1024`, copied unchanged.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q12_launcher` launches `q12_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = min((n + 1023) / 1024, 8192)`, when `n > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- The five columns: rows of the join of `lineitem` (ship mode MAIL or SHIP) with `orders`, `n_elements = shipmode.numel()` rows each, at least 1 at the launch.
- `out_high`, `out_low`: `torch::zeros({2})`, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
- `shipmode_ptr`: `(l_shipmode == shipmode2).astype("int32")`, so each value is 0 or 1, the ship mode's slot in `out_high` and `out_low`: `index-in` into `out_high` (2 elements). `priority_ptr` holds 0 or 1 and is used in a comparison.
- The receipt date bounds are any values.
