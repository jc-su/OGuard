# q11_core_cuda_reduce_by_key_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q11/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `reduce_by_key_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <cuda.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the other kernels of the source: `filter_kernel`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`reduce_by_key_launcher` launches `reduce_by_key_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = min(1024, (n + BLOCK_SIZE - 1) / BLOCK_SIZE)`, `BLOCK_SIZE = 1024`, and no dynamic shared memory, after `unique_count.fill_(0)`; `launch.json` states the block and the grid claim.

## Lengths and ranges

- `keys_sorted`: `n` elements: `ps_partkey` of the joined frame in sorted order, `n = keys_sorted.numel()` rows.
- `values_sorted`: `n` elements: `ps_supplycost * ps_availqty` in the same order.
- `keys_out`: `torch.zeros(n_rows)`: `n` elements; the kernel writes one position per run of equal keys, at most `n` in all.
- `values_out`: `n` elements, as `keys_out`.
- `unique_count`: `torch.zeros(1, int32)`, and `fill_(0)` before the launch; the kernel takes positions from it with `atomicAdd`.
- `n`: `n = keys_sorted.numel()`, the size symbol of the row buffers.
