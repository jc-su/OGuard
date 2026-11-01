# q17_cuda_q17_pass2_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q17/round_02.py`, copied as `plan.py`: `q17_pass2_kernel` in `_CUDA_SOURCE`, lines 38-90 of the plan file.

## Extraction edits

- `__global__ void q17_pass2_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses. The kernel uses `#define BLOCK_SIZE 1024`, copied unchanged.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q17_launcher` launches `q17_pass2_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = (n + 1023) / 1024`, after `q17_pass1_kernel`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_partkey`, `l_quantity`, `l_extendedprice`: columns of `lineitem`, `n_lineitems` rows each, at least 1.
- `part_map`: `max_partkey + 1` elements, holding -1 or an index in `[0, num_parts)`.
- `sum_qty`, `cnt_qty`: the `num_parts` sums and counts of the previous launch.
- `out_total`: `torch::zeros({1})`, zeroed and accumulated with `atomicAdd`: readwrite.
