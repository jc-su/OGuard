# q05_core_cuda_q5_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q05/round_01.py`, copied as `plan.py`: `q5_kernel` in `_CUDA_SOURCE`, lines 14-56 of the plan file.

## Extraction edits

- `__global__ void q5_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q5_launcher` launches `q5_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>` with `BLOCK_SIZE = 1024`, `blocks = (n + 1023) / 1024`, `num_copies = min(8, 49152 / (num_nations * 8))` (at least 1), and `shared_mem_size = num_copies * num_nations * 8` bytes of dynamic shared memory, when `n > 0` and `num_nations > 0`.

The dynamic shared memory depends on `num_nations`. This corpus kernel is the launch with `num_nations <= 768`, where the plan keeps `num_copies = 8` and passes `64 * num_nations` bytes. The plan's TPC-H launch has `num_nations = 5` (the nations of ASIA). With more than 768 nations the plan passes `num_copies = 49152 / (8 * num_nations)` copies, a shared memory that is not affine in `num_nations`; that launch is not in the corpus.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 64 * num_nations bytes, with the size maximum `num_nations <= 768`.

## Lengths and ranges

- `nation_id`, `extendedprice`, `discount`: columns of the joined frame, `n_elements = l_extendedprice.numel()` rows each, at least 1 at the launch.
- `out`: `torch::zeros({num_nations})`, zeroed by the plan before the launch and accumulated with `atomicAdd`: readwrite; `num_nations = len(nations_unique)` is its length (size), at least 1 at the launch and at most 768 in this configuration.
- `num_copies = 8` in this configuration.
- `nation_id` holds the row numbers `range(len(nations_unique))` joined by `n_nationkey`, values in `[0, num_nations)`, but the kernel checks each value, so it is `read`.
