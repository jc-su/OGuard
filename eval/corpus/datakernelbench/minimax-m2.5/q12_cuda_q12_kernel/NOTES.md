# q12_cuda_q12_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q12/round_10.py`, copied as `plan.py`.

## Extraction

- The kernel `q12_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cstdint>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q12_launcher` launches `q12_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `l_commitdate`: `n_elements` elements: `l_commitdate.numel()`, one element per joined row.
- `l_receiptdate`: `n_elements` elements: one element per joined row.
- `l_shipdate`: `n_elements` elements: one element per joined row.
- `l_shipmode`: `n_elements` elements: one element per joined row; the plan fills it with `(l_shipmode == shipmode2).astype(int32)`, 0 or 1, which the signature cannot state.
- `o_orderpriority_high`: `n_elements` elements: one element per joined row; 0 or 1 by the plan's cast of a boolean, which the signature cannot state.
- `receipt_lo_days`: a date bound, days since 1970-01-01 that the plan computes from its `receipt_lo` parameter; any int32 value; not used in an address.
- `receipt_hi_days`: a date bound, days since 1970-01-01 that the plan computes from its `receipt_hi` parameter; any int32 value; not used in an address.
- `out`: `torch::zeros({4}, kInt32)`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n_elements`: `n = l_commitdate.numel()`, the size symbol of the row buffers.
