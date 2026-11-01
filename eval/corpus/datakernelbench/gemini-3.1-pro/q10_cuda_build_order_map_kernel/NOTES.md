# q10_cuda_build_order_map_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q10/round_02.py`, copied as `plan.py`: `build_order_map_kernel` in `_CUDA_SOURCE`, lines 14-32 of the plan file.

## Extraction edits

- `__global__ void build_order_map_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q10_launcher` launches `build_order_map_kernel<<<blocks_orders, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks_orders = (n_orders + 1023) / 1024`, when `n_orders > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `o_orderkey`, `o_custkey`, `o_orderdate`: columns of `orders`, `n = o_orderkey.numel()` rows each, at least 1 at the launch.
- `order_map`: `torch::full({max_orderkey + 1}, -1)`, filled by the plan before the launch, with `max_orderkey = int(df_orders["o_orderkey"].max())`: readwrite.
- `o_orderkey`: the plan sizes `order_map` by the maximum of this column, and the TPC-H order keys are positive, so every value lies in `[0, max_orderkey]`: `index-in` into `order_map`.
- The date bounds are any values.
