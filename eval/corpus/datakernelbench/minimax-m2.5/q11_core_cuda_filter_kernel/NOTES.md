# q11_core_cuda_filter_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q11/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `filter_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <cuda.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the other kernels of the source: `reduce_by_key_kernel`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`filter_launcher` launches `filter_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n_in + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory, after `out_count.fill_(0)`; `launch.json` states the block and the grid claim.

## Lengths and ranges

- `keys_in`: `n_in` elements: `unique_keys_out[:actual_unique]`, the distinct part keys.
- `values_in`: `n_in` elements: the per-key sums in the same order.
- `keys_out`: `torch.zeros(actual_unique)`: `n_in` elements; the kernel writes one position per row above the threshold, at most `n_in` in all.
- `values_out`: `n_in` elements, as `keys_out`.
- `out_count`: `torch.zeros(1, int32)`, and `fill_(0)` before the launch; the kernel takes positions from it with `atomicAdd`.
- `n_in`: `keys_in.numel()` as int32, the size symbol of the row buffers.
- `threshold`: `total * threshold_fraction`; not used in an address.
