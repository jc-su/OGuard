# q17_cuda_q17_pass1_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q17/round_02.py`, copied as `plan.py`: `q17_pass1_kernel` in `_CUDA_SOURCE`, lines 13-36 of the plan file.

## Extraction edits

- `__global__ void q17_pass1_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q17_launcher` launches `q17_pass1_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks = (n + 1023) / 1024`; it returns before the launch when `n == 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_partkey`, `l_quantity`: columns of `lineitem`, `n_lineitems = l_partkey.numel()` rows each, at least 1.
- `part_map`: `torch.full((max_partkey + 1,), -1)` with `max_partkey` the largest key of the filtered `part` rows, and `part_map[part_keys] = arange(num_parts)`; `max_partkey` is the argument of its length (size).
- `sum_qty`, `cnt_qty`: `torch::zeros({num_parts})`, `num_parts` the filtered part rows, at least 1 (the plan returns earlier when there are none), zeroed and accumulated with `atomicAdd`: readwrite.
- `l_partkey` values are not bounded by the filtered parts' maximum (the kernel checks them): `read`. `part_map` holds -1, so it is not an index buffer: `read`.
