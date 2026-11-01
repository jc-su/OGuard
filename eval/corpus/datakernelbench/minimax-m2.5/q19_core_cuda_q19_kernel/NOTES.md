# q19_core_cuda_q19_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q19/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `q19_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Copied the macros that the kernel uses: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q19_launcher` launches `q19_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `l_quantity`: `n` elements: one element per row of the lineitem-part join, `n = l_quantity.numel()`.
- `l_extendedprice`: `n` elements: one element per joined row.
- `l_discount`: `n` elements: one element per joined row.
- `p_brand_id`: `n` elements: the brand code 12, 23, 34, or 0, one element per joined row.
- `p_container_grp`: `n` elements: the container group 1, 2, 3, or 0, one element per joined row.
- `p_size`: `n` elements: one element per joined row.
- `brand12_id`, `brand23_id`, `brand34_id`: the constants 12, 23, and 34; not used in an address.
- `sm_grp`, `md_grp`, `lg_grp`: the constants 1, 2, and 3; not used in an address.
- `out`: `torch::zeros({1})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n`: `n = l_quantity.numel()`, the size symbol of the row buffers.
