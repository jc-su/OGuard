# q20_core_cuda_build_supp_hash_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q20/round_03.py`, copied as `plan.py`: `build_supp_hash_kernel` in `_CUDA_SOURCE`, lines 72-93 of the plan file.

## Extraction edits

- `__global__ void build_supp_hash_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- `#include <cstdint>` is added: the kernel uses `uint64_t`, which the plan's `<torch/extension.h>` declares through `<cstdint>`; the front end compiles without the PyTorch include path, and the CUDA headers that nvcc includes do not declare it.
- `#include <cuda_runtime.h>` is left out: nvcc includes it for every CUDA source.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q20_launcher` launches `build_supp_hash_kernel<<<blocks_ps, threads>>>` with `threads = 256` and `blocks_ps = (N_ps + 255) / 256`, when `blocks_ps > 0`, after `probe_and_agg_kernel`. The kernel uses the device function `hash_single` (lines 20-26), copied unchanged.

`launch.json` states the block (256, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `ps_sum_qty`: the `N_ps` sums of the previous launch; `ps_availqty`, `ps_suppkey`: columns of the filtered `partsupp`, `N_ps` rows each, at least 1 at the launch.
- `supp_hash_set`: `torch::full({H_s}, -1)` of int64, passed as `unsigned long long int*`, filled by the plan before the launch and claimed with `atomicCAS`: readwrite; `H_s` is its length (size), the least power of two of at least `2 * N_ps` and at least 1024.
