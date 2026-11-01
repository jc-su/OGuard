# q09_cuda_q9_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q09/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `q9_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q9_aggregate` launches `q9_kernel<<<grid, block_size>>>` with `grid = (n + block_size - 1) / block_size`, `block_size = BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `price`: `n` elements: `price.numel()`, one element per row of the joined frame.
- `discount`: `n` elements: one element per joined row.
- `supplycost`: `n` elements: one element per joined row.
- `quantity`: `n` elements: one element per joined row.
- `nation_key`: `n` elements: one element per joined row, the supplier's `n_nationkey`.
- `year`: `n` elements: one element per joined row, the year of `o_orderdate`.
- `out`: `torch.zeros(num_groups)` with `num_groups = num_nations * year_range` and `num_nations = len(df_nation)`, which is 25 at every TPC-H scale factor; zeroed before the launch, and the kernel adds to it with `atomicAdd`.
- `n`: `price.numel()`; the plan returns before the launch when the join is empty, so its least value is 1.
- `num_nations`: the plan passes the constant 25.
- `year_min`: `int(year_arr.min())`, the least year of the same rows; not bounded in the signature.
- `year_range`: `year_max - year_min + 1` from the same years, the size symbol of `out` with the factor 25; at least 1.
