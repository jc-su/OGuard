# q13_core_cuda_cust_dist_kernel_optimized

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q13/round_01.py`, copied as `plan.py`: `cust_dist_kernel_optimized` in `_CUDA_SOURCE`, lines 27-59 of the plan file.

## Extraction edits

- `__global__ void cust_dist_kernel_optimized(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q13_launcher` launches `cust_dist_kernel_optimized<<<blocks_cust, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks_cust = (n_customers + 1023) / 1024`, when `blocks_cust > 0`, after `count_orders_kernel`. The kernel uses `#define HIST_BINS 1024`, copied unchanged.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `c_custkey`: the `c_custkey` column of `customer`, `n_customers = c_custkey.numel()` rows, at least 1 at the launch.
- `order_counts`: the `num_customers + 1` counts of the previous launch, with `num_customers = int(df_customer["c_custkey"].max())`.
- `count_frequencies`: `torch::zeros({n_orders + 1})` with `n_orders` the filtered orders' row count, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
- `c_custkey`: the plan sizes `order_counts` by the maximum of this column, and the TPC-H customer keys are positive: `index-in` into `order_counts`.
