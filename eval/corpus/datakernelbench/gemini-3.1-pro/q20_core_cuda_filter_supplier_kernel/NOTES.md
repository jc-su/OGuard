# q20_core_cuda_filter_supplier_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q20/round_03.py`, copied as `plan.py`: `filter_supplier_kernel` in `_CUDA_SOURCE`, lines 95-121 of the plan file.

## Extraction edits

- `__global__ void filter_supplier_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- `#include <cstdint>` is added: the kernel uses `uint64_t`, which the plan's `<torch/extension.h>` declares through `<cstdint>`; the front end compiles without the PyTorch include path, and the CUDA headers that nvcc includes do not declare it.
- `#include <cuda_runtime.h>` is left out: nvcc includes it for every CUDA source.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q20_launcher` launches `filter_supplier_kernel<<<blocks_s, threads>>>` with `threads = 256` and `blocks_s = (N_s + 255) / 256`, when `blocks_s > 0`, after `build_supp_hash_kernel`. The kernel uses the device function `hash_single` (lines 20-26), copied unchanged.

`launch.json` states the grid maxima (16777216, 1, 1): `2^32 / 256` blocks in x, where the unsigned 32-bit row index wraps and a row would run twice, and 1 in y and z, which the kernel ignores.

`launch.json` states the block (256, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `s_suppkey`, `s_row_idx`: the `s_suppkey` column and the row numbers `range(len(supplier_in_nation))` of the nation's suppliers, `N_s = s_suppkey.numel()` rows each, at least 1 at the launch.
- `supp_hash_set`: the `H_s` slots of the previous launch; `H_s` is the argument of its length (size), at least 1024.
- `out_idx`: `torch::empty({N_s})`, written at the positions the counter gives: write, `N_s` elements.
- `out_count`: `torch::zeros({1})`, zeroed by the plan and advanced with `atomicAdd`: readwrite.
