# q05_cuda_q5_revenue_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q05/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `q5_revenue_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Copied the macros that the kernel uses: `#define MAX_NATIONS 25`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q5_launcher` launches `q5_revenue_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and `shared_mem = num_nations * sizeof(double)` bytes of dynamic shared memory, which the kernel does not use (its `block_agg` array is static); `launch.json` states both.

## Lengths and ranges

- `nation_ids`: `n_rows` elements, the rows of `final_df`. Each value is a `nation_id` that the plan assigns as `range(len(nations_unique))` and joins back on `n_nationkey`, so it lies in [0, num_nations): an index into `out`.
- `price`: `n_rows` elements: one element per row of `final_df`.
- `discount`: `n_rows` elements: one element per row of `final_df`.
- `out`: `torch::zeros({num_nations})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n_rows`: `n = nation_ids.numel()`; the launcher returns before the launch when it is 0, so its least value is 1.
- `num_nations`: `len(nations_unique)`, the nations of one region of the TPC-H nation table, which has 25 rows at every scale factor: `launch.json` states the maximum 25. At least 1, since `final_df` has rows and each row has a nation.
