# q13_cuda_hist_counts_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q13/round_04.py`, copied as `plan.py`: `hist_counts_kernel` in `_CUDA_SOURCE`, lines 34-77 of the plan file.

## Extraction edits

- `__global__ void hist_counts_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses. The kernel uses `#define MAX_HIST_BINS 1024`, copied unchanged.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q13_launcher` launches `hist_counts_kernel<<<blocks_c, 1024>>>` with `blocks_c = (n_customers + 1023) / 1024`, when `n_customers > 0`, after `count_orders_kernel`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `c_custkey`: the `c_custkey` column of `customer`, `n = c_custkey.numel()` rows, at least 1 at the launch.
- `counts`: the `max_custkey + 1` counts of the previous launch, with `max_custkey = int(df_customer["c_custkey"].max())`.
- `c_custkey`: the plan sizes `counts` by the maximum of this column, and the TPC-H customer keys are positive: `index-in` into `counts`.
- `hist`: `torch::zeros({hist_size})` with `hist_size = 100000`, zeroed by the plan and accumulated with `atomicAdd`: readwrite, 100000 elements; `hist_size` is 100000.
