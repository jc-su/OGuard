# q18_core_cuda_q18_agg_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q18/round_02.py`, copied as `plan.py`: `q18_agg_kernel` in `_CUDA_SOURCE`, lines 13-29 of the plan file.

## Extraction edits

- `__global__ void q18_agg_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- `#include <cstdint>` is added: the kernel uses `uint8_t`, which the plan's `<torch/extension.h>` declares through `<cstdint>`; the front end compiles without the PyTorch include path, and the CUDA headers that nvcc includes do not declare it.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q18_launcher` launches `q18_agg_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = (n + 1023) / 1024`, when `n > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `keys`, `values`: the `l_orderkey` and `l_quantity` columns of `lineitem`, `n = keys.numel()` rows each, at least 1 at the launch.
- `sums`, `exists`: `torch.zeros(M)` with `M = max_val + 1` and `max_val = int(l_orderkey.max())`, zeroed by the plan before the launch and accumulated with `atomicAdd` or set: readwrite.
- `keys`: the plan sizes `sums` and `exists` by this column's maximum, and the TPC-H order keys are positive: `index-in` into `sums`.
