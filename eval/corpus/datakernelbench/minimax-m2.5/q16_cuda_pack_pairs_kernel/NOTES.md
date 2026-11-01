# q16_cuda_pack_pairs_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q16/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `pack_pairs_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cstdint>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <algorithm>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 256`.
- Left out the host code (the launchers and the pybind11 module).
- Left out the other kernels of the source: `count_unique_atomic_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`count_distinct_launcher` launches `pack_pairs_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 256`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `group_ids`: `n` elements: `group_ids.numel()`, one element per joined row.
- `supp_ids`: `n` elements: one element per joined row; the launcher checks that its length is `n`.
- `packed`: `n` elements: `torch::empty({n})`.
- `n`: `n = group_ids.numel()`; the plan returns before the launch when the join is empty, so its least value is 1.
