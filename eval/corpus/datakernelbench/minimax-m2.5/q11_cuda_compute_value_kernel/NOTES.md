# q11_cuda_compute_value_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q11/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `compute_value_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`compute_value_launcher` launches `compute_value_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `partkey_in`: `n` elements: `partkey_in.numel()`, one element per joined partsupp row.
- `supplycost`: `n` elements: one element per joined row.
- `availqty`: `n` elements: one element per joined row.
- `partkey_out`: `n` elements: `torch::empty({n})`.
- `value_out`: `n` elements: `torch::empty({n})`.
- `n`: `n = partkey_in.numel()`, the size symbol of every buffer.
