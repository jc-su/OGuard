# q16_cuda_count_unique_atomic_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q16/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `count_unique_atomic_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cstdint>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <algorithm>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 256`.
- Left out the host code (the launchers and the pybind11 module).
- Left out the other kernels of the source: `pack_pairs_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`count_distinct_launcher` launches `count_unique_atomic_kernel<<<blocks, BLOCK_SIZE>>>` after sorting the packed pairs, with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 256`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `sorted_packed_pairs`: `num_pairs` elements: `sorted`, the sorted output of `pack_pairs_kernel`, one element per joined row; each value holds a group id in its high 32 bits, which the signature cannot state.
- `group_counts`: `torch::zeros({num_groups})`, one element per (brand, type, size) group, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `num_pairs`: `n = group_ids.numel()`, at least 1.
