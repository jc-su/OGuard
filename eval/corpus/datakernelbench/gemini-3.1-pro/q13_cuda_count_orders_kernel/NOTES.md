# q13_cuda_count_orders_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q13/round_04.py`, copied as `plan.py`: `count_orders_kernel` in `_CUDA_SOURCE`, lines 12-32 of the plan file.

## Extraction edits

- `__global__ void count_orders_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q13_launcher` launches `count_orders_kernel<<<blocks_o, 1024>>>` with `blocks_o = (n_orders + 1023) / 1024`, when `n_orders > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `o_custkey`: the `o_custkey` column of `orders`; `mask`: the `o_comment` filter of the same rows; `n = o_custkey.numel()` rows each, at least 1 at the launch.
- `counts`: `torch::zeros({max_custkey + 1})` with `max_custkey = int(df_customer["c_custkey"].max())`, zeroed by the plan and accumulated with `atomicAdd`: readwrite; `max_custkey` is the argument (size).
- `o_custkey` refers to `customer` by a foreign key, which the plan does not check (the kernel does): `read`.
