# q02_cuda_filter_part_size

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q02/round_02.py`, copied as `plan.py`: `filter_part_size` in `_CUDA_SOURCE`, lines 11-21 of the plan file.

## Extraction edits

- `__global__ void filter_part_size(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`filter_part_size_launcher` launches `filter_part_size<<<blocks, 256>>>` with `blocks = (n + 255) / 256`, when `n > 0`.

`launch.json` states the block (256, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `p_size`: the `p_size` column of `part` as int64, `n = p_size.numel()` rows.
- `out_mask`: `torch::empty({n})` of bool, which the kernel writes: write.
- `n`: at least 1 at the launch; `size_val` is any value.
