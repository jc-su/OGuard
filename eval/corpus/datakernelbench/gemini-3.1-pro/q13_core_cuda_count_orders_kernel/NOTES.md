# q13_core_cuda_count_orders_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q13/round_01.py`, copied as `plan.py`: `count_orders_kernel` in `_CUDA_SOURCE`, lines 13-25 of the plan file.

## Extraction edits

- `__global__ void count_orders_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q13_launcher` launches `count_orders_kernel<<<blocks_orders, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks_orders = (n_orders + 1023) / 1024`, when `blocks_orders > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `o_custkey`: the `o_custkey` column of the orders whose comment passes the filter, `n_orders = o_custkey_filtered.numel()` rows, at least 1 at the launch.
- `order_counts`: `torch::zeros({num_customers + 1})` with `num_customers = int(df_customer["c_custkey"].max())`, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
- `o_custkey` refers to `customer` by a foreign key, which neither the plan nor the kernel checks: `read`.
