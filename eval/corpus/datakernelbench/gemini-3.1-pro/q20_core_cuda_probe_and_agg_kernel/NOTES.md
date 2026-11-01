# q20_core_cuda_probe_and_agg_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q20/round_03.py`, copied as `plan.py`: `probe_and_agg_kernel` in `_CUDA_SOURCE`, lines 46-70 of the plan file.

## Extraction edits

- `__global__ void probe_and_agg_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- `#include <cstdint>` is added: the kernel uses `uint64_t`, which the plan's `<torch/extension.h>` declares through `<cstdint>`; the front end compiles without the PyTorch include path, and the CUDA headers that nvcc includes do not declare it.
- `#include <cuda_runtime.h>` is left out: nvcc includes it for every CUDA source.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q20_launcher` launches `probe_and_agg_kernel<<<blocks_l, threads>>>` with `threads = 256` and `blocks_l = (N_l + 255) / 256`, when `blocks_l > 0`, after `build_ps_hash_kernel`. The kernel uses the device function `hash_pair` (lines 12-18), copied unchanged.

`launch.json` states the block (256, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_partkey`, `l_suppkey`, `l_quantity`: columns of the filtered `lineitem`, `N_l = l_partkey.numel()` rows each, at least 1 at the launch.
- `ps_partkey`, `ps_suppkey`: columns of the filtered `partsupp`, `N_ps` rows each, possibly none.
- `hash_table`: the `H` slots of the previous launch, holding -1 or a row index below `N_ps`; `H` is the argument of its length (size), at least 1024.
- `ps_sum_qty`: `torch::zeros({N_ps})`, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
- `hash_table` holds -1, so it is not an index buffer: `read`.
