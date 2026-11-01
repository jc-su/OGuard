# q22_core_cuda_ker_avg

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q22/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `ker_avg` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Copied the macros that the kernel uses: `#define BLOCK_SIZE 1024`.
- Left out the macros and device functions that the kernel does not use: `#define MAX_CODE 100`, `exists_key`.
- Left out the other kernels of the source: `ker_grp`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`cuda_avg` launches `ker_avg<<<(a.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `b`: `n` elements: `c_acctbal`, one element per customer row, `n = a.numel()`.
- `m`: `n` elements: the country-code membership mask, one element per customer row.
- `s`: `torch::zeros({1})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `c`: `torch::zeros({1}, kInt64)`, as `s`.
- `n`: `a.numel()`, the size symbol of the row buffers.
