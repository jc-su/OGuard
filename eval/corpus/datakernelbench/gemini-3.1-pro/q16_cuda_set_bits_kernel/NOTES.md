# q16_cuda_set_bits_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q16/round_02.py`, copied as `plan.py`: `set_bits_kernel` in `_CUDA_SOURCE`, lines 13-32 of the plan file.

## Extraction edits

- `__global__ void set_bits_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- `#include <cstdint>` is added: the kernel uses `uint32_t`, which the plan's `<torch/extension.h>` declares through `<cstdint>`; the front end compiles without the PyTorch include path, and the CUDA headers that nvcc includes do not declare it.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q16_count_distinct_launcher` launches `set_bits_kernel<<<blocks_set, BLOCK_SIZE>>>` with `BLOCK_SIZE = 256` and `blocks_set = (n + 255) / 256`, when `blocks_set > 0`.

`launch.json` states the block (256, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `group_ids`, `supp_ids`: columns of the join, `n_elements = group_ids.numel()` rows each, at least 1 (the plan returns before the launch when the join is empty).
- `bitmap`: `torch::zeros({num_words, num_groups})` of int32, passed as `uint32_t*`, with `num_words = (max_supp_id + 32) / 32` and `max_supp_id = supp_ids.max()`, zeroed by the plan and updated with `atomicOr`: readwrite. Its length `num_words * num_groups` is not affine in the sizes, so its length is the size symbol `bitmap`.
- `num_groups = len(group_df)`, at least 1 (the plan returns before the launch when it is 0).
- `group_ids` lie in `[0, num_groups)` and `supp_ids` in `[0, max_supp_id]`, but the kernel's buffers have other lengths: `read`.
