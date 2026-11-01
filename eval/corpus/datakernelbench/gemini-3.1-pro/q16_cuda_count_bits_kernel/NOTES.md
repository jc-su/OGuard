# q16_cuda_count_bits_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q16/round_02.py`, copied as `plan.py`: `count_bits_kernel` in `_CUDA_SOURCE`, lines 34-49 of the plan file.

## Extraction edits

- `__global__ void count_bits_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- `#include <cstdint>` is added: the kernel uses `uint32_t`, which the plan's `<torch/extension.h>` declares through `<cstdint>`; the front end compiles without the PyTorch include path, and the CUDA headers that nvcc includes do not declare it.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q16_count_distinct_launcher` launches `count_bits_kernel<<<blocks_count, BLOCK_SIZE>>>` with `BLOCK_SIZE = 256` and `blocks_count = (num_groups + 255) / 256`, when `blocks_count > 0`, after `set_bits_kernel`.

`launch.json` states the block (256, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `bitmap`: the `num_words * num_groups` words of the previous launch, whose length is the size symbol `bitmap` (not affine in the sizes).
- `counts`: `torch::zeros({num_groups})` of int64, zeroed by the plan before the launch: readwrite; `num_groups` is its length (size), at least 1.
- `num_words = (max_supp_id + 32) / 32`, at least 1.
