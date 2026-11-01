# q18_cuda_filter_big_groups

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q18/round_04.py`, copied as `plan.py`.

## Extraction

- The kernel `filter_big_groups` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <thrust/sort.h>`, `#include <thrust/reduce.h>`, `#include <thrust/copy.h>`, `#include <thrust/device_vector.h>`, `#include <thrust/device_ptr.h>`, `#include <thrust/iterator/zip_iterator.h>`, `#include <thrust/functional.h>`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q18_launcher` launches `filter_big_groups<<<grid, BLOCK_SIZE>>>` after Thrust's `sort_by_key` and `reduce_by_key`, with `grid = (unique_count + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 256`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `keys`: `n` elements: `unique_keys_vec`, resized to the `unique_count` distinct order keys.
- `sums`: `n` elements: `unique_sums_vec`, resized to `unique_count`.
- `out_keys`: `n` elements: `big_keys_vec(unique_count)`.
- `out_sums`: `n` elements: `big_sums_vec(unique_count)`.
- `threshold`: the qty_threshold parameter, 300.0; not used in an address.
- `n`: `unique_count`, the size symbol of all four buffers.
- `count_out`: `big_count_vec(1)`, set to 0 before the launch; the kernel takes positions from it with `atomicAdd`.
