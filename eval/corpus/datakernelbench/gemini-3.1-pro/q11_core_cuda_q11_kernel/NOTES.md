# q11_core_cuda_q11_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q11/round_01.py`, copied as `plan.py`: `q11_kernel` in `_CUDA_SOURCE`, lines 13-45 of the plan file.

## Extraction edits

- `__global__ void q11_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q11_launcher` launches `q11_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = (n + 1023) / 1024`, when `n > 0`. The kernel uses `#define BLOCK_SIZE 1024`, copied unchanged.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `partkey`, `supplycost`, `availqty`: columns of the joined frame, `n_elements = partkey.numel()` rows each, at least 1 at the launch.
- `grouped_values`: `torch::zeros({max_key + 1})` with `max_key = int(ps_partkey.max())`, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
- `total_sum`: `torch::zeros({1})`, zeroed and accumulated with `atomicAdd`: readwrite.
- `partkey`: the plan sizes `grouped_values` by this column's maximum, and the TPC-H part keys are positive: `index-in` into `grouped_values`.
